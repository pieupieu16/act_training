"""Low-cost color cue for the single green box in the current pick/place scene.

This is a task-specific cue, not a general object detector. It uses no GPU and
can be run identically during training and deployment.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class GreenBoxConfig:
    hue_min: int = 55
    hue_max: int = 94
    saturation_min: int = 150
    value_min: int = 40
    roi_top_fraction: float = 0.45
    roi_right_fraction: float = 0.88
    min_component_area: int = 80


def green_box_mask(rgb: np.ndarray, config: GreenBoxConfig = GreenBoxConfig()) -> np.ndarray:
    """Return uint8 mask (0/1) in the original RGB image coordinates."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("green_box_mask cần ảnh RGB uint8 [H,W,3].")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(
        hsv,
        (config.hue_min, config.saturation_min, config.value_min),
        (config.hue_max, 255, 255),
    )
    h, w = mask.shape
    mask[:int(h * config.roi_top_fraction)] = 0
    mask[:, int(w * config.roi_right_fraction):] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    kept = np.zeros((h, w), dtype=np.uint8)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= config.min_component_area:
            kept[labels == label] = 1
    return kept


def green_box_bbox(rgb: np.ndarray, config: GreenBoxConfig = GreenBoxConfig()) -> Optional[Tuple[int, int, int, int]]:
    """Return largest visible green component as x1,y1,x2,y2, or None."""
    mask = green_box_mask(rgb, config)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[label]
    return int(x), int(y), int(x + w), int(y + h)
