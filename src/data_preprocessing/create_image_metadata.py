#!/usr/bin/env python3
"""Create image-level metadata from germination dataset filenames.

Usage:
  python create_image_metadata.py .
"""

from __future__ import annotations

import argparse
import csv
import re
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


# Ví dụ tên file: zm1_10_img001.jpg
# prefix = zm
# experiment_id = 1
# dish_id = 10
# frame_id = 001
FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>[a-z]+)(?P<experiment_id>\d+)_(?P<dish_id>\d+)_img(?P<frame_id>\d+)$",
    re.IGNORECASE,
)


def find_subdir(root: Path, name: str) -> Optional[Path]:
    candidate = root / name
    return candidate if candidate.exists() and candidate.is_dir() else None


def parse_filename(filename_stem: str) -> Optional[Dict[str, object]]:
    match = FILENAME_PATTERN.match(filename_stem)

    if not match:
        return None

    prefix = match.group("prefix").lower()
    experiment_id = int(match.group("experiment_id"))
    dish_id = int(match.group("dish_id"))
    frame_id = int(match.group("frame_id"))

    sequence_id = f"{prefix}{experiment_id}_{dish_id}"

    return {
        "file_stem": filename_stem,
        "species_code_from_filename": prefix,
        "experiment_id": experiment_id,
        "dish_id": dish_id,
        "frame_id": frame_id,
        "sequence_id": sequence_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create image-level metadata from dataset filenames."
    )

    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Dataset root folder, for example: DL/",
    )

    parser.add_argument(
        "--output",
        default="image_metadata.csv",
        help="Output CSV file name.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    if not root.exists() or not root.is_dir():
        print(f"Root folder not found: {root}")
        return 1

    rows = []
    parse_errors = []

    species_counter = Counter()
    sequence_counter = Counter()
    frame_counter = Counter()

    for plant_folder in sorted(root.iterdir()):
        if not plant_folder.is_dir():
            continue

        if plant_folder.name not in SPECIES_BY_FOLDER:
            continue

        img_dir = find_subdir(plant_folder, "img")
        ann_dir = find_subdir(plant_folder, "true_ann")

        if img_dir is None:
            print(f"Missing img/ folder in {plant_folder.name}")
            continue

        if ann_dir is None:
            print(f"Missing true_ann/ folder in {plant_folder.name}")
            continue

        species_info = SPECIES_BY_FOLDER[plant_folder.name]

        image_files = sorted(
            path for path in img_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"Scanning {plant_folder.name}: {len(image_files)} images")

        for image_path in image_files:
            parsed = parse_filename(image_path.stem)

            if parsed is None:
                parse_errors.append(str(image_path))
                continue

            xml_path = ann_dir / f"{image_path.stem}.xml"
            has_xml = xml_path.exists()

            species_code_from_folder = species_info["species_code"]
            species_code_from_filename = parsed["species_code_from_filename"]

            prefix_match = species_code_from_folder == species_code_from_filename

            row = {
                "image_path": str(image_path),
                "xml_path": str(xml_path),
                "filename": image_path.name,
                "file_stem": image_path.stem,
                "folder_name": plant_folder.name,
                "species_code": species_code_from_folder,
                "species_name": species_info["species_name"],
                "vietnamese_name": species_info["vietnamese_name"],
                "species_code_from_filename": species_code_from_filename,
                "prefix_match": prefix_match,
                "experiment_id": parsed["experiment_id"],
                "dish_id": parsed["dish_id"],
                "frame_id": parsed["frame_id"],
                "sequence_id": parsed["sequence_id"],
                "has_xml": has_xml,
            }

            rows.append(row)

            species_counter[species_info["species_name"]] += 1
            sequence_counter[parsed["sequence_id"]] += 1
            frame_counter[parsed["frame_id"]] += 1

    if not rows:
        print("No metadata rows created.")
        return 1

    output_path = root / args.output

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
        "prefix_match",
        "experiment_id",
        "dish_id",
        "frame_id",
        "sequence_id",
        "has_xml",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== METADATA SUMMARY =====")
    print(f"Total images: {len(rows)}")
    print(f"Total sequences: {len(sequence_counter)}")
    print(f"Filename parse errors: {len(parse_errors)}")

    print("\nImages by species:")
    for species_name, count in species_counter.items():
        print(f"  {species_name}: {count}")

    print("\nSequences by species code:")
    sequence_by_prefix = Counter(seq[:2] for seq in sequence_counter.keys())

    for prefix, count in sorted(sequence_by_prefix.items()):
        print(f"  {prefix}: {count}")

    prefix_mismatch_count = sum(1 for row in rows if not row["prefix_match"])
    missing_xml_count = sum(1 for row in rows if not row["has_xml"])

    print("\nQuality checks:")
    print(f"  Prefix mismatch: {prefix_mismatch_count}")
    print(f"  Missing XML according to metadata: {missing_xml_count}")

    if parse_errors:
        print("\nFiles with filename parse errors:")
        for path in parse_errors[:20]:
            print(f"  {path}")

    print(f"\nSaved metadata to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())