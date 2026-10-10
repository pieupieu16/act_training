#!/usr/bin/env python3
"""Save a contact sheet for reviewing the CPU green-box mask on an episode."""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

from act_pipeline.data.object_mask import green_box_mask, green_box_bbox


def main():
    parser = argparse.ArgumentParser(description="Preview object mask on an HDF5 episode")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", default="outputs/object_mask_preview.jpg")
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples phải >= 1")

    with h5py.File(args.episode, "r") as f:
        images = f["observations/images/chest_rgb"]
        indices = np.linspace(0, len(images) - 1, args.samples, dtype=int)
        tiles = []
        for index in indices:
            rgb = images[index]
            mask = green_box_mask(rgb)
            box = green_box_bbox(rgb)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr[mask > 0] = (0.5 * bgr[mask > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            if box is not None:
                cv2.rectangle(bgr, box[:2], box[2:], (0, 255, 255), 2)
            tile = cv2.resize(bgr, (424, 240))
            cv2.putText(tile, str(index), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            tiles.append(tile)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise OSError(f"Không thể ghi {output}")
    print(f"Đã lưu {output}")


if __name__ == "__main__":
    main()
