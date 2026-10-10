#!/usr/bin/env python3
"""Export RGB frames from ACT HDF5 episodes for one-class YOLO annotation."""

import argparse
import glob
import json
from pathlib import Path

import cv2
import h5py
import yaml

from act_pipeline.data.object_mask import green_box_mask, green_box_bbox


def main():
    parser = argparse.ArgumentParser(description="Export RGB images while preserving episode train/val split")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_json", help="episode_split.json from ACT run")
    parser.add_argument("--stride", type=int, default=10, help="Export one frame every N timesteps")
    parser.add_argument("--pseudo_boxes", action="store_true",
                        help="Chỉ xuất frame có mask hộp xanh rõ và tạo nhãn box YOLO để review")
    parser.add_argument("--min_mask_pixels", type=int, default=1500)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride phải >= 1")

    files = sorted(glob.glob(str(Path(args.dataset_dir) / "*.hdf5")) +
                   glob.glob(str(Path(args.dataset_dir) / "*.h5")))
    if not files:
        parser.error("Không tìm thấy HDF5 episodes.")
    files = [str(Path(x).resolve()) for x in files]
    if args.split_json:
        with open(args.split_json, "r", encoding="utf-8") as f:
            split = json.load(f)
        train_files = {str(Path(x).resolve()) for x in split["train"]}
        val_files = {str(Path(x).resolve()) for x in split["val"]}
        if train_files & val_files or train_files | val_files != set(files):
            parser.error("split_json không khớp với dataset_dir.")
    else:
        # Deterministic split when an ACT split is not available yet.
        import random
        shuffled = files.copy()
        random.Random(42).shuffle(shuffled)
        val_files = set(shuffled[:max(1, int(len(files) * 0.15))]) if len(files) >= 2 else set()
        train_files = set(files) - val_files

    output = Path(args.output_dir).resolve()
    for subset in ("train", "val"):
        (output / "images" / subset).mkdir(parents=True, exist_ok=True)
        (output / "labels" / subset).mkdir(parents=True, exist_ok=True)

    manifest = []
    for path in files:
        subset = "val" if path in val_files else "train"
        with h5py.File(path, "r") as f:
            images = f["observations/images"]
            key = "chest_rgb" if "chest_rgb" in images else "chest"
            frames = images[key]
            for index in range(0, len(frames), args.stride):
                rgb = frames[index][..., :3]
                bbox = None
                if args.pseudo_boxes:
                    mask = green_box_mask(rgb)
                    bbox = green_box_bbox(rgb)
                    if bbox is None:
                        continue
                    x1, y1, x2, y2 = bbox
                    if int(mask[y1:y2, x1:x2].sum()) < args.min_mask_pixels:
                        continue
                name = f"{Path(path).stem}_{index:06d}.jpg"
                target = output / "images" / subset / name
                if not cv2.imwrite(str(target), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    raise OSError(f"Không thể ghi {target}")
                if bbox is not None:
                    h, w = rgb.shape[:2]
                    x1, y1, x2, y2 = bbox
                    # Include box edges that are not green, such as the blue label.
                    pad_x, pad_y = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
                    x1, x2 = max(0, x1 - pad_x), min(w, x2 + pad_x)
                    y1, y2 = max(0, y1 - pad_y), min(h, y2 + pad_y)
                    label = output / "labels" / subset / f"{Path(name).stem}.txt"
                    label.write_text(f"0 {(x1 + x2) / (2*w):.6f} {(y1 + y2) / (2*h):.6f} "
                                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n",
                                     encoding="utf-8")
                manifest.append({"image": str(target), "episode": path,
                                 "frame": index, "split": subset,
                                 "pseudo_box": bbox is not None})

    with (output / "frame_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with (output / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"path": str(output), "train": "images/train",
                        "val": "images/val", "names": {0: "plastic_box"}},
                       f, sort_keys=False, allow_unicode=True)
    print(f"Đã xuất {len(manifest)} ảnh vào {output}.")
    if args.pseudo_boxes:
        print("Nhãn box tự sinh chỉ là gợi ý; review và sửa trước khi train YOLO.")
    if not val_files:
        print("Mới có một episode: chưa có tập val độc lập để train detector.")


if __name__ == "__main__":
    main()
