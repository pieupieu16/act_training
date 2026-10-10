"""Asynchronous HDF5 recorder for operator-corrected robot episodes.

The action stored in /action is the command actually approved/executed by the
operator. /correction/policy_action stores the policy's proposed command.
"""

import os
import queue
import threading
from pathlib import Path

import h5py
import numpy as np


class CorrectionEpisodeRecorder:
    def __init__(self, path: str, image_shape=(240, 424), queue_size: int = 64):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.partial_path = Path(str(self.path) + ".partial")
        if self.partial_path.exists():
            raise FileExistsError(self.partial_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.image_shape = tuple(image_shape)
        self.queue = queue.Queue(maxsize=queue_size)
        self.error = None
        self.closed = False
        self.last_timestamp_ns = -1
        self.worker = threading.Thread(target=self._write, daemon=True)
        self.worker.start()

    def append(self, rgb, depth, qpos, action, policy_action, timestamp_ns, intervention: bool):
        if self.closed:
            raise RuntimeError("Recorder đã đóng.")
        if self.error is not None:
            raise RuntimeError("Ghi correction episode thất bại.") from self.error
        rgb = np.asarray(rgb)
        depth = np.asarray(depth)
        qpos = np.asarray(qpos)
        action = np.asarray(action)
        policy_action = np.asarray(policy_action)
        if rgb.shape != (*self.image_shape, 3) or rgb.dtype != np.uint8:
            raise ValueError("RGB phải là uint8 [H,W,3].")
        if depth.shape != self.image_shape or depth.dtype != np.uint16:
            raise ValueError("Depth phải là uint16 [H,W] và đã align với RGB.")
        if any(x.shape != (16,) or not np.isfinite(x).all() for x in (qpos, action, policy_action)):
            raise ValueError("qpos/action/policy_action phải là vector hữu hạn 16 chiều.")
        timestamp_ns = int(timestamp_ns)
        if timestamp_ns <= self.last_timestamp_ns:
            raise ValueError("timestamp_ns phải tăng nghiêm ngặt.")
        item = (rgb.copy(), depth.copy(), qpos.astype(np.float32).copy(),
                action.astype(np.float32).copy(), policy_action.astype(np.float32).copy(),
                timestamp_ns, bool(intervention))
        try:
            self.queue.put_nowait(item)
        except queue.Full as exc:
            raise RuntimeError("Hàng đợi ghi HDF5 đầy; dừng episode để không mất frame.") from exc
        self.last_timestamp_ns = timestamp_ns

    def close(self):
        if self.closed:
            return
        self.closed = True
        while self.worker.is_alive():
            try:
                self.queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self.worker.join()
        if self.error is not None:
            raise RuntimeError(f"Ghi correction episode thất bại: {self.partial_path}") from self.error
        os.replace(self.partial_path, self.path)

    def _write(self):
        h, w = self.image_shape
        try:
            with h5py.File(self.partial_path, "w") as f:
                f.attrs["sim"] = False
                f.attrs["robot_type"] = "OpenArm_Bimanual_16DOF"
                f.attrs["num_joints"] = 16
                f.attrs["depth_scale"] = 0.001
                f.attrs["source"] = "operator_correction"
                f.require_group("observations/images")
                f.require_group("correction")
                datasets = {
                    "observations/images/chest_rgb": f.create_dataset("observations/images/chest_rgb", (0, h, w, 3), maxshape=(None, h, w, 3), chunks=(1, h, w, 3), dtype="u1", compression="lzf"),
                    "observations/images/chest_depth": f.create_dataset("observations/images/chest_depth", (0, h, w), maxshape=(None, h, w), chunks=(1, h, w), dtype="u2", compression="lzf"),
                    "observations/qpos": f.create_dataset("observations/qpos", (0, 16), maxshape=(None, 16), dtype="f4"),
                    "action": f.create_dataset("action", (0, 16), maxshape=(None, 16), dtype="f4"),
                    "correction/policy_action": f.create_dataset("correction/policy_action", (0, 16), maxshape=(None, 16), dtype="f4"),
                    "timestamp_ns": f.create_dataset("timestamp_ns", (0,), maxshape=(None,), dtype="i8"),
                    "correction/intervention": f.create_dataset("correction/intervention", (0,), maxshape=(None,), dtype="?"),
                }
                count = 0
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    for dataset, value in zip(datasets.values(), item):
                        dataset.resize(count + 1, axis=0)
                        dataset[count] = value
                    count += 1
                if count == 0:
                    raise ValueError("Episode rỗng; không tạo HDF5.")
                f.attrs["frequency_hz"] = 50.0
        except Exception as exc:
            self.error = exc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
