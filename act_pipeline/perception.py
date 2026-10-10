"""Optional low-rate object localization from a one-class YOLO model and aligned depth."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class ObjectObservation:
    pixel: Tuple[float, float]
    xyz_camera_m: Tuple[float, float, float]
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]


class SingleObjectPerception:
    """Returns camera-frame position; robot-frame transform belongs to calibration."""

    def __init__(self, weights: str, intrinsics: CameraIntrinsics, confidence: float = 0.4,
                 imgsz: int = 416, device: str = "cpu", depth_scale_m: float = 0.001,
                 depth_range_m: Tuple[float, float] = (0.2, 1.2)):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Cài detector: pip install -e '.[detector]'") from exc
        self.model = YOLO(weights)
        self.intrinsics = intrinsics
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device
        self.depth_scale_m = depth_scale_m
        self.depth_range_m = depth_range_m
        if intrinsics.fx <= 0 or intrinsics.fy <= 0 or depth_scale_m <= 0:
            raise ValueError("Camera intrinsics và depth_scale_m phải dương.")
        if depth_range_m[0] <= 0 or depth_range_m[0] >= depth_range_m[1]:
            raise ValueError("depth_range_m không hợp lệ.")

    def locate(self, rgb: np.ndarray, aligned_depth: np.ndarray) -> Optional[ObjectObservation]:
        """RGB uint8 and aligned depth uint16 must have identical H,W."""
        if rgb.ndim != 3 or rgb.shape[2] != 3 or aligned_depth.shape != rgb.shape[:2]:
            raise ValueError("Cần RGB [H,W,3] và depth đã align [H,W].")
        if rgb.dtype != np.uint8 or aligned_depth.dtype != np.uint16:
            raise ValueError("Cần RGB uint8 và depth uint16 theo DATA_FORMAT_SPECIFICATION.md.")

        # Ultralytics takes numpy images in BGR order.
        result = self.model.predict(rgb[..., ::-1].copy(), conf=self.confidence,
                                    imgsz=self.imgsz, device=self.device,
                                    retina_masks=True, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None
        index = int(result.boxes.conf.argmax().item())
        box = result.boxes.xyxy[index].cpu().numpy()
        h, w = aligned_depth.shape
        valid = np.zeros((h, w), dtype=bool)
        if result.masks is not None and index < len(result.masks.data):
            import cv2
            mask = result.masks.data[index].cpu().numpy().astype(np.uint8)
            if mask.shape != (h, w):
                mask = np.zeros((h, w), dtype=np.uint8)
                polygon = result.masks.xy[index].astype(np.int32)
                if len(polygon) >= 3:
                    cv2.fillPoly(mask, [polygon], 1)
            valid = mask.astype(bool)
        else:
            x1, y1, x2, y2 = box.astype(int)
            valid[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True

        depth_m = aligned_depth.astype(np.float32) * self.depth_scale_m
        valid &= (depth_m >= self.depth_range_m[0]) & (depth_m <= self.depth_range_m[1])
        if not np.any(valid):
            return None
        ys, xs = np.nonzero(valid)
        u = float(np.median(xs))
        v = float(np.median(ys))
        z = float(np.median(depth_m[valid]))
        if not np.isfinite(z) or z <= 0:
            return None
        k = self.intrinsics
        return ObjectObservation(
            pixel=(u, v),
            xyz_camera_m=((u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z),
            confidence=float(result.boxes.conf[index].item()),
            bbox_xyxy=tuple(float(x) for x in box),
        )
