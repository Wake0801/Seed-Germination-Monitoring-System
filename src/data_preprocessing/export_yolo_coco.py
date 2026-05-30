#!/usr/bin/env python3
"""Export object-level metadata to YOLO and/or COCO format.

Input:
  object_metadata_with_split.csv

Output example:
  germination_binary/
  ├── images/
  │   ├── train/
  │   ├── val/
  │   └── test/
  ├── labels/
  │   ├── train/
  │   ├── val/
  │   └── test/
  ├── annotations/
  │   ├── instances_train.json
  │   ├── instances_val.json
  │   └── instances_test.json
  ├── data.yaml
  └── export_summary.txt

Usage:
  python export_yolo_coco.py . --label-mode binary --format both
  python export_yolo_coco.py . --label-mode six_class --format both
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


SPLITS = ["train", "val", "test"]


def get_class_config(label_mode: str) -> Tuple[str, str, List[str]]:
    """Return class_id column, label column, and class names."""

    if label_mode == "binary":
        class_id_col = "binary_class_id"
        label_col = "binary_label"
        class_names = [
            "non-germinated",
            "germinated",
        ]
        return class_id_col, label_col, class_names

    if label_mode == "six_class":
        class_id_col = "six_class_id"
        label_col = "six_class_label"
        class_names = [
            "pg_im",
            "pg_el",
            "sc_im",
            "sc_el",
            "zm_im",
            "zm_el",
        ]
        return class_id_col, label_col, class_names

    raise ValueError(f"Unsupported label mode: {label_mode}")


def resolve_path(root: Path, value: str) -> Path:
    """Resolve image path from CSV. Handles absolute and relative paths."""

    path = Path(str(value))

    if path.is_absolute():
        return path

    return root / path


def ensure_dirs(output_root: Path) -> None:
    for split in SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    (output_root / "annotations").mkdir(parents=True, exist_ok=True)


def compute_yolo_bbox(row: pd.Series) -> Tuple[float, float, float, float]:
    """Compute normalized YOLO bbox: x_center, y_center, width, height."""

    image_width = float(row["image_width"])
    image_height = float(row["image_height"])

    xmin = float(row["xmin"])
    ymin = float(row["ymin"])
    xmax = float(row["xmax"])
    ymax = float(row["ymax"])

    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    x_center = xmin + bbox_width / 2.0
    y_center = ymin + bbox_height / 2.0

    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    width_norm = bbox_width / image_width
    height_norm = bbox_height / image_height

    return x_center_norm, y_center_norm, width_norm, height_norm


def validate_yolo_bbox(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> bool:
    return (
        0.0 <= x_center <= 1.0
        and 0.0 <= y_center <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
    )


def copy_images(df: pd.DataFrame, root: Path, output_root: Path) -> None:
    """Copy images into images/train, images/val, images/test."""

    image_df = (
        df[
            [
                "image_path",
                "filename",
                "split",
                "image_width",
                "image_height",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    duplicated = image_df.duplicated(subset=["split", "filename"], keep=False)

    if duplicated.any():
        duplicates = image_df[duplicated][["split", "filename", "image_path"]]
        print("Duplicated image filename detected inside the same split:")
        print(duplicates.head(20))
        raise RuntimeError("Duplicated filenames would overwrite images.")

    print("\nCopying images:")

    for split in SPLITS:
        split_images = image_df[image_df["split"] == split]
        print(f"  {split}: {len(split_images)} images")

        for _, row in split_images.iterrows():
            src = resolve_path(root, row["image_path"])
            dst = output_root / "images" / split / Path(row["filename"]).name

            if not src.exists():
                raise FileNotFoundError(f"Image not found: {src}")

            shutil.copy2(src, dst)


def export_yolo(
    df: pd.DataFrame,
    output_root: Path,
    class_id_col: str,
    class_names: List[str],
) -> Dict[str, int]:
    """Export YOLO .txt label files."""

    print("\nExporting YOLO labels:")

    stats = {}

    for split in SPLITS:
        split_df = df[df["split"] == split]
        label_dir = output_root / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)

        num_label_files = 0
        num_objects = 0
        num_errors = 0

        grouped = split_df.groupby("filename", sort=True)

        for filename, group in grouped:
            label_path = label_dir / f"{Path(filename).stem}.txt"
            lines = []

            for _, row in group.iterrows():
                class_id = int(row[class_id_col])

                if class_id < 0 or class_id >= len(class_names):
                    num_errors += 1
                    continue

                x_center, y_center, width, height = compute_yolo_bbox(row)

                if not validate_yolo_bbox(x_center, y_center, width, height):
                    num_errors += 1
                    continue

                line = (
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}"
                )

                lines.append(line)
                num_objects += 1

            with label_path.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines))
                if lines:
                    f.write("\n")

            num_label_files += 1

        stats[f"{split}_label_files"] = num_label_files
        stats[f"{split}_objects"] = num_objects
        stats[f"{split}_errors"] = num_errors

        print(
            f"  {split}: "
            f"{num_label_files} label files, "
            f"{num_objects} objects, "
            f"{num_errors} skipped"
        )

    return stats


def write_data_yaml(
    output_root: Path,
    class_names: List[str],
) -> None:
    """Write Ultralytics YOLO data.yaml."""

    data_yaml_path = output_root / "data.yaml"

    lines = []
    lines.append(f"path: {output_root.as_posix()}")
    lines.append("train: images/train")
    lines.append("val: images/val")
    lines.append("test: images/test")
    lines.append("")
    lines.append(f"nc: {len(class_names)}")
    lines.append("names:")

    for class_id, class_name in enumerate(class_names):
        lines.append(f"  {class_id}: {class_name}")

    with data_yaml_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    print(f"\nSaved YOLO data.yaml: {data_yaml_path}")


def export_coco(
    df: pd.DataFrame,
    output_root: Path,
    class_id_col: str,
    class_names: List[str],
) -> Dict[str, int]:
    """Export COCO JSON annotations.

    Note:
      YOLO class ids start from 0.
      COCO category ids are exported from 1 to N.
    """

    print("\nExporting COCO annotations:")

    categories = [
        {
            "id": class_id + 1,
            "name": class_name,
            "supercategory": "seed",
        }
        for class_id, class_name in enumerate(class_names)
    ]

    stats = {}

    for split in SPLITS:
        split_df = df[df["split"] == split]

        images = []
        annotations = []

        image_id_map = {}
        annotation_id = 1

        image_groups = split_df.groupby("filename", sort=True)

        for image_id, (filename, group) in enumerate(image_groups, start=1):
            first = group.iloc[0]

            image_id_map[filename] = image_id

            images.append(
                {
                    "id": image_id,
                    "file_name": Path(filename).name,
                    "width": int(first["image_width"]),
                    "height": int(first["image_height"]),
                }
            )

            for _, row in group.iterrows():
                class_id = int(row[class_id_col])

                if class_id < 0 or class_id >= len(class_names):
                    continue

                xmin = float(row["xmin"])
                ymin = float(row["ymin"])
                xmax = float(row["xmax"])
                ymax = float(row["ymax"])

                bbox_width = xmax - xmin
                bbox_height = ymax - ymin

                if bbox_width <= 0 or bbox_height <= 0:
                    continue

                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id + 1,
                        "bbox": [
                            xmin,
                            ymin,
                            bbox_width,
                            bbox_height,
                        ],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0,
                    }
                )

                annotation_id += 1

        coco_data = {
            "info": {
                "description": "Seed germination dataset exported from Pascal VOC metadata",
                "version": "1.0",
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        output_json = output_root / "annotations" / f"instances_{split}.json"

        with output_json.open("w", encoding="utf-8") as f:
            json.dump(coco_data, f, ensure_ascii=False)

        stats[f"{split}_images"] = len(images)
        stats[f"{split}_annotations"] = len(annotations)

        print(
            f"  {split}: "
            f"{len(images)} images, "
            f"{len(annotations)} annotations -> {output_json.name}"
        )

    return stats


def write_summary(
    df: pd.DataFrame,
    output_root: Path,
    label_col: str,
    class_id_col: str,
    class_names: List[str],
    export_format: str,
    label_mode: str,
) -> None:
    summary_path = output_root / "export_summary.txt"

    lines = []

    lines.append("EXPORT SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Label mode: {label_mode}")
    lines.append(f"Export format: {export_format}")
    lines.append(f"Number of classes: {len(class_names)}")
    lines.append("")
    lines.append("Class names:")
    for class_id, class_name in enumerate(class_names):
        lines.append(f"  {class_id}: {class_name}")

    lines.append("")
    lines.append("Dataset size:")
    lines.append(f"  Total object rows: {len(df)}")
    lines.append(f"  Total images: {df['filename'].nunique()}")
    lines.append(f"  Total sequences: {df['sequence_id'].nunique()}")

    lines.append("")
    lines.append("Images by split:")
    lines.append(
        str(
            df[["split", "filename"]]
            .drop_duplicates()
            .groupby("split")
            .size()
        )
    )

    lines.append("")
    lines.append("Objects by split:")
    lines.append(str(df.groupby("split").size()))

    lines.append("")
    lines.append("Objects by split and label:")
    lines.append(str(df.groupby(["split", label_col]).size().unstack(fill_value=0)))

    lines.append("")
    lines.append("Objects by split and species:")
    lines.append(str(df.groupby(["split", "species_code"]).size().unstack(fill_value=0)))

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    print(f"\nSaved export summary: {summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export object metadata to YOLO and/or COCO format."
    )

    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Dataset root folder containing object_metadata_with_split.csv",
    )

    parser.add_argument(
        "--input",
        default="object_metadata_with_split.csv",
        help="Input CSV file.",
    )

    parser.add_argument(
        "--label-mode",
        choices=["binary", "six_class"],
        default="binary",
        help="Use binary labels or six-class labels.",
    )

    parser.add_argument(
        "--format",
        choices=["yolo", "coco", "both"],
        default="both",
        help="Export YOLO, COCO, or both.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output folder. Default: germination_<label_mode>",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    input_path = root / args.input

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    class_id_col, label_col, class_names = get_class_config(args.label_mode)

    output_name = args.output or f"germination_{args.label_mode}"
    output_root = root / output_name

    output_root.mkdir(parents=True, exist_ok=True)
    ensure_dirs(output_root)

    df = pd.read_csv(input_path)

    required_columns = {
        "image_path",
        "filename",
        "split",
        "sequence_id",
        "species_code",
        "image_width",
        "image_height",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        class_id_col,
        label_col,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        print("Missing required columns:")
        for col in sorted(missing_columns):
            print(f"  - {col}")
        return 1

    expected_splits = set(SPLITS)
    actual_splits = set(df["split"].unique())

    if actual_splits != expected_splits:
        print(f"Unexpected split values: {actual_splits}")
        print(f"Expected: {expected_splits}")
        return 1

    print("Loaded split metadata:")
    print(f"  Total object rows: {len(df)}")
    print(f"  Total images: {df['filename'].nunique()}")
    print(f"  Total sequences: {df['sequence_id'].nunique()}")
    print(f"  Label mode: {args.label_mode}")
    print(f"  Export format: {args.format}")
    print(f"  Output folder: {output_root}")

    print("\nClass mapping:")
    for class_id, class_name in enumerate(class_names):
        print(f"  {class_id}: {class_name}")

    copy_images(df, root, output_root)

    if args.format in {"yolo", "both"}:
        export_yolo(df, output_root, class_id_col, class_names)
        write_data_yaml(output_root, class_names)

    if args.format in {"coco", "both"}:
        export_coco(df, output_root, class_id_col, class_names)

    write_summary(
        df=df,
        output_root=output_root,
        label_col=label_col,
        class_id_col=class_id_col,
        class_names=class_names,
        export_format=args.format,
        label_mode=args.label_mode,
    )

    print("\nDone.")
    print(f"Exported dataset folder: {output_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())