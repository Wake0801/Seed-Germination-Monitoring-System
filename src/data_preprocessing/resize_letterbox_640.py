#!/usr/bin/env python3
"""Resize/letterbox YOLO/COCO dataset images to 640x640.

Input:
  germination_binary/
  ├── images/train|val|test
  ├── labels/train|val|test
  ├── annotations/instances_train.json ...
  └── data.yaml

Output:
  germination_binary_640/
  ├── images/train|val|test
  ├── labels/train|val|test
  ├── annotations/instances_train.json ...
  ├── data.yaml
  └── letterbox_summary.txt

Usage:
  python resize_letterbox_640.py . --input germination_binary --output germination_binary_640 --size 640 --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image


SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def letterbox_image(
    image: Image.Image,
    target_size: int = 640,
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, float, int, int, int, int]:
    """Resize image with unchanged aspect ratio and pad to target_size x target_size.

    Returns:
      output_image, scale_ratio, pad_left, pad_top, new_width, new_height
    """

    original_width, original_height = image.size

    ratio = min(
        target_size / original_width,
        target_size / original_height,
    )

    new_width = int(round(original_width * ratio))
    new_height = int(round(original_height * ratio))

    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR,
    )

    canvas = Image.new("RGB", (target_size, target_size), pad_color)

    pad_left = (target_size - new_width) // 2
    pad_top = (target_size - new_height) // 2

    canvas.paste(resized, (pad_left, pad_top))

    return canvas, ratio, pad_left, pad_top, new_width, new_height


def transform_yolo_label_line(
    line: str,
    original_width: int,
    original_height: int,
    target_size: int,
    ratio: float,
    pad_left: int,
    pad_top: int,
) -> str:
    """Transform one YOLO label line after letterbox.

    YOLO input:
      class_id x_center_norm y_center_norm width_norm height_norm

    Output is also normalized by target_size.
    """

    parts = line.strip().split()

    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO label line: {line}")

    class_id = parts[0]
    x_center_norm = float(parts[1])
    y_center_norm = float(parts[2])
    width_norm = float(parts[3])
    height_norm = float(parts[4])

    # Convert normalized coordinates to original pixel coordinates
    x_center = x_center_norm * original_width
    y_center = y_center_norm * original_height
    bbox_width = width_norm * original_width
    bbox_height = height_norm * original_height

    # Apply resize ratio and padding
    new_x_center = x_center * ratio + pad_left
    new_y_center = y_center * ratio + pad_top
    new_bbox_width = bbox_width * ratio
    new_bbox_height = bbox_height * ratio

    # Normalize by target size
    new_x_center_norm = new_x_center / target_size
    new_y_center_norm = new_y_center / target_size
    new_width_norm = new_bbox_width / target_size
    new_height_norm = new_bbox_height / target_size

    # Numeric safety for tiny floating-point errors
    new_x_center_norm = min(max(new_x_center_norm, 0.0), 1.0)
    new_y_center_norm = min(max(new_y_center_norm, 0.0), 1.0)
    new_width_norm = min(max(new_width_norm, 0.0), 1.0)
    new_height_norm = min(max(new_height_norm, 0.0), 1.0)

    return (
        f"{class_id} "
        f"{new_x_center_norm:.6f} "
        f"{new_y_center_norm:.6f} "
        f"{new_width_norm:.6f} "
        f"{new_height_norm:.6f}"
    )


def process_yolo_labels(
    input_label_path: Path,
    output_label_path: Path,
    original_width: int,
    original_height: int,
    target_size: int,
    ratio: float,
    pad_left: int,
    pad_top: int,
) -> int:
    """Transform YOLO label file and return number of objects."""

    if not input_label_path.exists():
        # In object detection, empty label files are allowed for negative images.
        # But your dataset should have labels for every image.
        output_label_path.write_text("", encoding="utf-8")
        return 0

    lines = [
        line.strip()
        for line in input_label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    transformed_lines = []

    for line in lines:
        transformed_lines.append(
            transform_yolo_label_line(
                line=line,
                original_width=original_width,
                original_height=original_height,
                target_size=target_size,
                ratio=ratio,
                pad_left=pad_left,
                pad_top=pad_top,
            )
        )

    output_label_path.write_text(
        "\n".join(transformed_lines) + ("\n" if transformed_lines else ""),
        encoding="utf-8",
    )

    return len(transformed_lines)


def process_images_and_yolo_labels(
    input_root: Path,
    output_root: Path,
    target_size: int,
) -> Dict[str, Dict[str, int]]:
    """Resize images and transform YOLO labels."""

    stats: Dict[str, Dict[str, int]] = {}

    for split in SPLITS:
        input_image_dir = input_root / "images" / split
        input_label_dir = input_root / "labels" / split

        output_image_dir = output_root / "images" / split
        output_label_dir = output_root / "labels" / split

        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            path
            for path in input_image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        num_images = 0
        num_objects = 0
        num_missing_labels = 0
        num_padded_images = 0

        print(f"\nProcessing split: {split}")
        print(f"  Found images: {len(image_files)}")

        for image_path in image_files:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                original_width, original_height = img.size

                output_img, ratio, pad_left, pad_top, new_width, new_height = letterbox_image(
                    image=img,
                    target_size=target_size,
                )

            if pad_left != 0 or pad_top != 0 or new_width != target_size or new_height != target_size:
                num_padded_images += 1

            output_image_path = output_image_dir / image_path.name
            output_img.save(output_image_path, quality=95)

            input_label_path = input_label_dir / f"{image_path.stem}.txt"
            output_label_path = output_label_dir / f"{image_path.stem}.txt"

            if not input_label_path.exists():
                num_missing_labels += 1

            object_count = process_yolo_labels(
                input_label_path=input_label_path,
                output_label_path=output_label_path,
                original_width=original_width,
                original_height=original_height,
                target_size=target_size,
                ratio=ratio,
                pad_left=pad_left,
                pad_top=pad_top,
            )

            num_images += 1
            num_objects += object_count

        stats[split] = {
            "images": num_images,
            "objects": num_objects,
            "missing_labels": num_missing_labels,
            "padded_images": num_padded_images,
        }

        print(f"  Saved images: {num_images}")
        print(f"  Saved YOLO objects: {num_objects}")
        print(f"  Missing labels: {num_missing_labels}")
        print(f"  Images with padding: {num_padded_images}")

    return stats


def transform_coco_bbox(
    bbox: list[float],
    ratio: float,
    pad_left: int,
    pad_top: int,
) -> list[float]:
    """Transform COCO bbox [x, y, width, height] after letterbox."""

    x, y, width, height = bbox

    new_x = x * ratio + pad_left
    new_y = y * ratio + pad_top
    new_width = width * ratio
    new_height = height * ratio

    return [
        round(new_x, 4),
        round(new_y, 4),
        round(new_width, 4),
        round(new_height, 4),
    ]


def process_coco_annotations(
    input_root: Path,
    output_root: Path,
    target_size: int,
) -> Dict[str, Dict[str, int]]:
    """Transform COCO JSON annotations to target_size."""

    input_ann_dir = input_root / "annotations"
    output_ann_dir = output_root / "annotations"
    output_ann_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Dict[str, int]] = {}

    for split in SPLITS:
        input_json_path = input_ann_dir / f"instances_{split}.json"
        output_json_path = output_ann_dir / f"instances_{split}.json"

        if not input_json_path.exists():
            print(f"\nCOCO file not found, skipping: {input_json_path}")
            continue

        with input_json_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        image_transform = {}

        for image_info in coco["images"]:
            old_width = int(image_info["width"])
            old_height = int(image_info["height"])

            ratio = min(target_size / old_width, target_size / old_height)
            new_width = int(round(old_width * ratio))
            new_height = int(round(old_height * ratio))

            pad_left = (target_size - new_width) // 2
            pad_top = (target_size - new_height) // 2

            image_transform[image_info["id"]] = {
                "ratio": ratio,
                "pad_left": pad_left,
                "pad_top": pad_top,
            }

            image_info["width"] = target_size
            image_info["height"] = target_size

        for ann in coco["annotations"]:
            transform = image_transform[ann["image_id"]]

            new_bbox = transform_coco_bbox(
                bbox=ann["bbox"],
                ratio=transform["ratio"],
                pad_left=transform["pad_left"],
                pad_top=transform["pad_top"],
            )

            ann["bbox"] = new_bbox
            ann["area"] = round(new_bbox[2] * new_bbox[3], 4)

        with output_json_path.open("w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)

        stats[split] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
        }

        print(f"\nSaved COCO {split}: {output_json_path}")
        print(f"  Images: {len(coco['images'])}")
        print(f"  Annotations: {len(coco['annotations'])}")

    return stats


def update_data_yaml(
    input_root: Path,
    output_root: Path,
) -> None:
    """Copy data.yaml and update path to output_root."""

    input_yaml = input_root / "data.yaml"
    output_yaml = output_root / "data.yaml"

    if not input_yaml.exists():
        print(f"\ndata.yaml not found, skipping: {input_yaml}")
        return

    lines = input_yaml.read_text(encoding="utf-8").splitlines()
    updated_lines = []

    for line in lines:
        if line.strip().startswith("path:"):
            updated_lines.append(f"path: {output_root.as_posix()}")
        else:
            updated_lines.append(line)

    output_yaml.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

    print(f"\nSaved updated data.yaml: {output_yaml}")


def write_summary(
    output_root: Path,
    target_size: int,
    yolo_stats: Dict[str, Dict[str, int]],
    coco_stats: Dict[str, Dict[str, int]],
) -> None:
    summary_path = output_root / "letterbox_summary.txt"

    lines = []
    lines.append("RESIZE / LETTERBOX SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Target size: {target_size}x{target_size}")
    lines.append("Pad color: (114, 114, 114)")
    lines.append("")
    lines.append("YOLO image/label summary:")

    for split in SPLITS:
        stat = yolo_stats.get(split, {})
        lines.append(
            f"  {split}: "
            f"images={stat.get('images', 0)}, "
            f"objects={stat.get('objects', 0)}, "
            f"missing_labels={stat.get('missing_labels', 0)}, "
            f"padded_images={stat.get('padded_images', 0)}"
        )

    lines.append("")
    lines.append("COCO summary:")

    for split in SPLITS:
        stat = coco_stats.get(split, {})
        lines.append(
            f"  {split}: "
            f"images={stat.get('images', 0)}, "
            f"annotations={stat.get('annotations', 0)}"
        )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved summary: {summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resize/letterbox YOLO/COCO dataset to 640x640."
    )

    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root folder containing germination_binary/",
    )

    parser.add_argument(
        "--input",
        default="germination_binary",
        help="Input exported dataset folder.",
    )

    parser.add_argument(
        "--output",
        default="germination_binary_640",
        help="Output resized dataset folder.",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=640,
        help="Target image size.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output folder if it already exists.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    input_root = root / args.input
    output_root = root / args.output

    if not input_root.exists():
        print(f"Input folder not found: {input_root}")
        return 1

    if output_root.exists():
        if args.overwrite:
            shutil.rmtree(output_root)
        else:
            print(f"Output folder already exists: {output_root}")
            print("Use --overwrite to recreate it.")
            return 1

    output_root.mkdir(parents=True, exist_ok=True)

    print("Resize/letterbox dataset")
    print(f"  Input: {input_root}")
    print(f"  Output: {output_root}")
    print(f"  Target size: {args.size}x{args.size}")

    yolo_stats = process_images_and_yolo_labels(
        input_root=input_root,
        output_root=output_root,
        target_size=args.size,
    )

    coco_stats = process_coco_annotations(
        input_root=input_root,
        output_root=output_root,
        target_size=args.size,
    )

    update_data_yaml(
        input_root=input_root,
        output_root=output_root,
    )

    write_summary(
        output_root=output_root,
        target_size=args.size,
        yolo_stats=yolo_stats,
        coco_stats=coco_stats,
    )

    print("\nDone.")
    print(f"Resized dataset folder: {output_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())