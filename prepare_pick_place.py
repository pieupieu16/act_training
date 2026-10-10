#!/usr/bin/env python3
"""Convert recorded HDF5 episodes into trainable ACT episodes.

The current recorder's /action duplicates /observations/qpos. Use the finite
/joint_commands samples as the target joint commands and discard only the
initial rows before commands first appear.
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np


def convert_episode(source: Path, target: Path, action_source: str) -> None:
    if target.exists() or Path(str(target) + ".partial").exists():
        raise FileExistsError(f"Output đã tồn tại: {target}")
    with h5py.File(source, "r") as src:
        qpos = src["observations/qpos"][:]
        actions = src[action_source][:]
        if qpos.shape != actions.shape or qpos.ndim != 2 or qpos.shape[1] != 16:
            raise ValueError(f"Shape qpos/action không hợp lệ trong {source}")
        finite = np.isfinite(actions).all(axis=1)
        if not finite.any():
            raise ValueError(f"Không có action hữu hạn trong {source}")
        first = int(np.argmax(finite))
        if not finite[first:].all():
            raise ValueError(f"Action có NaN sau đoạn khởi động trong {source}; kiểm tra thủ công.")
        if np.array_equal(qpos[first:], actions[first:]):
            raise ValueError(f"Action trùng toàn bộ qpos trong {source}; cần joint_commands thực.")
        timestamps = src["timestamp_ns"][first:]
        if len(timestamps) < 2 or not np.all(np.diff(timestamps) > 0):
            raise ValueError(f"Timestamp không tăng nghiêm ngặt trong {source}")

        target.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(str(target) + ".partial")
        try:
            with h5py.File(partial, "w", libver="latest") as dst:
                for key, value in src.attrs.items():
                    dst.attrs[key] = value
                dst.attrs["action_source"] = action_source
                dst.attrs["leading_frames_dropped"] = first
                dst.attrs["samples"] = len(timestamps)
                dst.attrs["duration_seconds"] = float((timestamps[-1] - timestamps[0]) / 1e9)

                dst.create_dataset("timestamp_ns", data=timestamps)
                obs = dst.create_group("observations")
                obs.create_dataset("qpos", data=qpos[first:].astype(np.float32))
                for optional in ("qvel", "effort"):
                    if optional in src["observations"]:
                        obs.create_dataset(optional, data=src[f"observations/{optional}"][first:])
                dst.create_dataset("action", data=actions[first:].astype(np.float32))

                images = obs.create_group("images")
                for key in ("chest_rgb", "chest_depth"):
                    source_ds = src[f"observations/images/{key}"]
                    if len(source_ds) != len(qpos):
                        raise ValueError(f"Độ dài ảnh {key} khác qpos trong {source}")
                    output_shape = (len(timestamps),) + source_ds.shape[1:]
                    output_ds = images.create_dataset(
                        key, shape=output_shape, dtype=source_ds.dtype,
                        chunks=(1,) + source_ds.shape[1:], compression="lzf",
                    )
                    for start in range(0, len(timestamps), 16):
                        end = min(start + 16, len(timestamps))
                        output_ds[start:end] = source_ds[first + start:first + end]
            os.replace(partial, target)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
    print(f"{source.name}: bỏ {first} frame đầu; lưu {len(timestamps)} frame vào {target}")


def main():
    parser = argparse.ArgumentParser(description="Prepare pick/place HDF5 episodes for ACT")
    parser.add_argument("--input", required=True, help="Một file HDF5 hoặc thư mục episodes")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--action_source", choices=("joint_commands", "action"), default="joint_commands")
    args = parser.parse_args()
    source = Path(args.input)
    files = [source] if source.is_file() else sorted(list(source.glob("*.hdf5")) + list(source.glob("*.h5")))
    if not files:
        parser.error("Không tìm thấy episode HDF5.")
    target_dir = Path(args.output_dir)
    for file in files:
        convert_episode(file, target_dir / file.name, args.action_source)


if __name__ == "__main__":
    main()
