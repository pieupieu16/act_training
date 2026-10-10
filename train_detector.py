#!/usr/bin/env python3
"""Train a one-class YOLO11n detector or segmenter from labeled images."""

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Train object perception for one-object pick and place")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML with train/val images and one class")
    parser.add_argument("--task", choices=("segment", "detect"), default="segment")
    parser.add_argument("--model", default=None, help="Starting weights; default yolo11n-seg.pt or yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="CUDA index or cpu")
    parser.add_argument("--project", default="checkpoints/detector")
    parser.add_argument("--name", default="pick_place_object")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.is_file():
        parser.error(f"Dataset YAML không tồn tại: {data_path}")
    with data_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "train" not in data or "val" not in data:
        parser.error("Dataset YAML cần có train và val.")
    names = data.get("names")
    if names is None or len(names) != 1:
        parser.error("Preset này yêu cầu đúng một class vật thể trong names.")
    root = Path(data.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    for subset in ("train", "val"):
        if data[subset] == f"images/{subset}":
            image_dir = root / "images" / subset
            label_dir = root / "labels" / subset
            if not any(image_dir.glob("*.jpg")) and not any(image_dir.glob("*.png")):
                parser.error(f"Thiếu ảnh {subset} trong {image_dir}; cần thêm episodes độc lập.")
            if not any(label_dir.glob("*.txt")):
                parser.error(f"Thiếu nhãn {subset} trong {label_dir}; review/gán nhãn trước khi train.")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Cài dependency detector: pip install -e '.[detector]'") from exc

    weights = args.model or ("yolo11n-seg.pt" if args.task == "segment" else "yolo11n.pt")
    model = YOLO(weights)
    model.train(
        data=str(data_path), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, project=args.project,
        name=args.name, exist_ok=False,
    )


if __name__ == "__main__":
    main()
