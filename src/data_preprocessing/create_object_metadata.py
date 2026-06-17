#!/usr/bin/env python3
"""Create object-level metadata from Pascal VOC XML annotations.

Each object/bounding box in XML becomes one row in object_metadata.csv.

Usage:
  python create_object_metadata.py .
"""

from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, Optional


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


SPECIES_BY_FOLDER = {
    "PennisetumGlaucum": {
        "species_code": "pg",
        "species_name": "Pennisetum glaucum",
        "vietnamese_name": "Kê ngọc trai",
    },
    "SecaleCereale": {
        "species_code": "sc",
        "species_name": "Secale cereale",
        "vietnamese_name": "Lúa mạch đen",
    },
    "ZeaMays": {
        "species_code": "zm",
        "species_name": "Zea mays",
        "vietnamese_name": "Ngô",
    },
}


# Ví dụ: zm1_10_img001
# species_code = zm
# experiment_id = 1
# dish_id = 10
# frame_id = 001
FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>[a-z]+)(?P<experiment_id>\d+)_(?P<dish_id>\d+)_img(?P<frame_id>\d+)$",
    re.IGNORECASE,
)


# Chuẩn hóa nhãn về bài toán 2 lớp
BINARY_LABEL_MAP = {
    "pg_im": ("non-germinated", 0),
    "sc_im": ("non-germinated", 0),
    "zm_im": ("non-germinated", 0),

    "pg_el": ("germinated", 1),
    "sc_el": ("germinated", 1),
    "zm_el": ("germinated", 1),
}


# Giữ nhãn 6 lớp nếu sau này muốn train model phân biệt cả loài + trạng thái
SIX_CLASS_ID_MAP = {
    "pg_im": 0,
    "pg_el": 1,
    "sc_im": 2,
    "sc_el": 3,
    "zm_im": 4,
    "zm_el": 5,
}


def find_subdir(root: Path, name: str) -> Optional[Path]:
    candidate = root / name
    return candidate if candidate.exists() and candidate.is_dir() else None


def text_of(elem: Optional[ET.Element]) -> str:
    return elem.text.strip() if elem is not None and elem.text else ""


def parse_filename(filename_stem: str) -> Optional[Dict[str, object]]:
    match = FILENAME_PATTERN.match(filename_stem)

    if not match:
        return None

    species_code = match.group("prefix").lower()
    experiment_id = int(match.group("experiment_id"))
    dish_id = int(match.group("dish_id"))
    frame_id = int(match.group("frame_id"))

    sequence_id = f"{species_code}{experiment_id}_{dish_id}"

    return {
        "species_code_from_filename": species_code,
        "experiment_id": experiment_id,
        "dish_id": dish_id,
        "frame_id": frame_id,
        "sequence_id": sequence_id,
    }


def read_xml_size(root: ET.Element) -> tuple[int, int, int]:
    size_node = root.find("size")

    if size_node is None:
        raise ValueError("Missing <size> node")

    width = int(float(text_of(size_node.find("width"))))
    height = int(float(text_of(size_node.find("height"))))

    depth_text = text_of(size_node.find("depth"))
    depth = int(float(depth_text)) if depth_text else 3

    return width, height, depth


def read_bbox(obj: ET.Element) -> tuple[int, int, int, int]:
    bndbox = obj.find("bndbox")

    if bndbox is None:
        raise ValueError("Missing <bndbox>")

    xmin = round(float(text_of(bndbox.find("xmin"))))
    ymin = round(float(text_of(bndbox.find("ymin"))))
    xmax = round(float(text_of(bndbox.find("xmax"))))
    ymax = round(float(text_of(bndbox.find("ymax"))))

    return xmin, ymin, xmax, ymax


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create object-level metadata from Pascal VOC XML annotations."
    )

    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Dataset root folder, for example: DL/",
    )

    parser.add_argument(
        "--output",
        default="object_metadata.csv",
        help="Output CSV file name.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    if not root.exists() or not root.is_dir():
        print(f"Root folder not found: {root}")
        return 1

    rows = []
    errors = []

    species_counter = Counter()
    raw_label_counter = Counter()
    binary_label_counter = Counter()
    six_class_counter = Counter()
    sequence_counter = Counter()
    objects_per_image_counter = Counter()

    for plant_folder in sorted(root.iterdir()):
        if not plant_folder.is_dir():
            continue

        if plant_folder.name not in SPECIES_BY_FOLDER:
            continue

        img_dir = find_subdir(plant_folder, "img")
        ann_dir = find_subdir(plant_folder, "true_ann")

        if img_dir is None or ann_dir is None:
            print(f"Skipping {plant_folder.name}: missing img/ or true_ann/")
            continue

        species_info = SPECIES_BY_FOLDER[plant_folder.name]

        xml_files = sorted(ann_dir.glob("*.xml"))

        print(f"Scanning {plant_folder.name}: {len(xml_files)} XML files")

        for xml_path in xml_files:
            file_stem = xml_path.stem

            parsed_name = parse_filename(file_stem)

            if parsed_name is None:
                errors.append({
                    "xml_path": str(xml_path),
                    "error": "filename_parse_error",
                    "detail": file_stem,
                })
                continue

            image_path = None

            for ext in IMAGE_EXTENSIONS:
                candidate = img_dir / f"{file_stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                errors.append({
                    "xml_path": str(xml_path),
                    "error": "missing_image",
                    "detail": file_stem,
                })
                continue

            try:
                tree = ET.parse(xml_path)
                xml_root = tree.getroot()
            except ET.ParseError as exc:
                errors.append({
                    "xml_path": str(xml_path),
                    "error": "xml_parse_error",
                    "detail": str(exc),
                })
                continue

            try:
                width, height, depth = read_xml_size(xml_root)
            except Exception as exc:
                errors.append({
                    "xml_path": str(xml_path),
                    "error": "size_read_error",
                    "detail": str(exc),
                })
                continue

            objects = xml_root.findall("object")

            if not objects:
                errors.append({
                    "xml_path": str(xml_path),
                    "error": "empty_annotation",
                    "detail": "XML has no object",
                })
                continue

            object_count_in_image = 0

            for object_id, obj in enumerate(objects):
                raw_label = text_of(obj.find("name"))

                if not raw_label:
                    errors.append({
                        "xml_path": str(xml_path),
                        "error": "missing_label",
                        "detail": f"object_id={object_id}",
                    })
                    continue

                if raw_label not in BINARY_LABEL_MAP:
                    errors.append({
                        "xml_path": str(xml_path),
                        "error": "unknown_raw_label",
                        "detail": raw_label,
                    })
                    continue

                try:
                    xmin, ymin, xmax, ymax = read_bbox(obj)
                except Exception as exc:
                    errors.append({
                        "xml_path": str(xml_path),
                        "error": "bbox_read_error",
                        "detail": f"object_id={object_id}, {exc}",
                    })
                    continue

                # Kiểm tra bbox hợp lệ
                if xmin >= xmax or ymin >= ymax:
                    errors.append({
                        "xml_path": str(xml_path),
                        "error": "invalid_bbox_order",
                        "detail": f"{raw_label}: {(xmin, ymin, xmax, ymax)}",
                    })
                    continue

                if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                    errors.append({
                        "xml_path": str(xml_path),
                        "error": "bbox_out_of_bounds",
                        "detail": f"{raw_label}: {(xmin, ymin, xmax, ymax)}, size={(width, height)}",
                    })
                    continue

                bbox_width = xmax - xmin
                bbox_height = ymax - ymin
                bbox_area = bbox_width * bbox_height

                x_center = xmin + bbox_width / 2
                y_center = ymin + bbox_height / 2

                # Tọa độ chuẩn hóa kiểu YOLO, tiện cho bước sau
                x_center_norm = x_center / width
                y_center_norm = y_center / height
                bbox_width_norm = bbox_width / width
                bbox_height_norm = bbox_height / height

                binary_label, binary_class_id = BINARY_LABEL_MAP[raw_label]
                six_class_label = raw_label
                six_class_id = SIX_CLASS_ID_MAP[raw_label]

                row = {
                    "image_path": str(image_path),
                    "xml_path": str(xml_path),
                    "filename": image_path.name,
                    "file_stem": file_stem,
                    "folder_name": plant_folder.name,

                    "species_code": species_info["species_code"],
                    "species_name": species_info["species_name"],
                    "vietnamese_name": species_info["vietnamese_name"],
                    "species_code_from_filename": parsed_name["species_code_from_filename"],

                    "experiment_id": parsed_name["experiment_id"],
                    "dish_id": parsed_name["dish_id"],
                    "frame_id": parsed_name["frame_id"],
                    "sequence_id": parsed_name["sequence_id"],

                    "image_width": width,
                    "image_height": height,
                    "image_depth": depth,

                    "object_id": object_id,
                    "raw_label": raw_label,

                    "binary_label": binary_label,
                    "binary_class_id": binary_class_id,

                    "six_class_label": six_class_label,
                    "six_class_id": six_class_id,

                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,

                    "bbox_width": bbox_width,
                    "bbox_height": bbox_height,
                    "bbox_area": bbox_area,

                    "x_center": x_center,
                    "y_center": y_center,

                    "x_center_norm": x_center_norm,
                    "y_center_norm": y_center_norm,
                    "bbox_width_norm": bbox_width_norm,
                    "bbox_height_norm": bbox_height_norm,
                }

                rows.append(row)

                object_count_in_image += 1

                species_counter[species_info["species_name"]] += 1
                raw_label_counter[raw_label] += 1
                binary_label_counter[binary_label] += 1
                six_class_counter[six_class_label] += 1
                sequence_counter[parsed_name["sequence_id"]] += 1

            objects_per_image_counter[object_count_in_image] += 1

    if not rows:
        print("No object metadata rows created.")
        return 1

    output_path = root / args.output
    error_path = root / "object_metadata_errors.csv"

    fieldnames = [
        "image_path",
        "xml_path",
        "filename",
        "file_stem",
        "folder_name",

        "species_code",
        "species_name",
        "vietnamese_name",
        "species_code_from_filename",

        "experiment_id",
        "dish_id",
        "frame_id",
        "sequence_id",

        "image_width",
        "image_height",
        "image_depth",

        "object_id",
        "raw_label",

        "binary_label",
        "binary_class_id",

        "six_class_label",
        "six_class_id",

        "xmin",
        "ymin",
        "xmax",
        "ymax",

        "bbox_width",
        "bbox_height",
        "bbox_area",

        "x_center",
        "y_center",

        "x_center_norm",
        "y_center_norm",
        "bbox_width_norm",
        "bbox_height_norm",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    error_fieldnames = ["xml_path", "error", "detail"]

    with error_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=error_fieldnames)
        writer.writeheader()
        writer.writerows(errors)

    print("\n===== OBJECT METADATA SUMMARY =====")
    print(f"Total objects / bounding boxes: {len(rows)}")
    print(f"Total sequences: {len(sequence_counter)}")
    print(f"Total errors skipped: {len(errors)}")

    print("\nObjects by species:")
    for species_name, count in species_counter.items():
        print(f"  {species_name}: {count}")

    print("\nRaw label distribution:")
    for label, count in sorted(raw_label_counter.items()):
        print(f"  {label}: {count}")

    print("\nBinary label distribution:")
    for label, count in sorted(binary_label_counter.items()):
        print(f"  {label}: {count}")

    print("\nSix-class label distribution:")
    for label, count in sorted(six_class_counter.items()):
        print(f"  {label}: {count}")

    print("\nObjects per image distribution:")
    for object_count, image_count in sorted(objects_per_image_counter.items()):
        print(f"  {object_count} objects/image: {image_count} images")

    print(f"\nSaved object metadata to: {output_path}")
    print(f"Saved skipped/error rows to: {error_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())