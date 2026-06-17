#!/usr/bin/env python3
"""Build crop-classification data from GermPredDataset.

This is the data pipeline used by the Custom CNN baseline:
  1. Read Pascal VOC XML files.
  2. Map raw labels to germinated / non_germinated.
  3. Split by sequence_id to avoid time-series leakage.
  4. Crop each seed bounding box into ImageFolder-style folders.

Usage:
  python src/data/build_crops.py
  python src/data/build_crops.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image

from split_by_sequence import (
    SPLITS,
    assign_splits,
    attach_split,
    check_leakage,
    make_sequence_rows,
    print_summary,
    write_rows,
)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

SPECIES_BY_FOLDER = {
    "PennisetumGlaucum": {
        "species_code": "pg",
        "species_name": "Pennisetum glaucum",
        "vietnamese_name": "Ke ngoc trai",
    },
    "SecaleCereale": {
        "species_code": "sc",
        "species_name": "Secale cereale",
        "vietnamese_name": "Lua mach den",
    },
    "ZeaMays": {
        "species_code": "zm",
        "species_name": "Zea mays",
        "vietnamese_name": "Ngo",
    },
}

LABEL_MAP = {
    "pg_im": ("non_germinated", 0),
    "sc_im": ("non_germinated", 0),
    "zm_im": ("non_germinated", 0),
    "pg_el": ("germinated", 1),
    "sc_el": ("germinated", 1),
    "zm_el": ("germinated", 1),
}

PROJECT_ROOT = Path.cwd().resolve()


def text_of(elem: Optional[ET.Element]) -> str:
    return elem.text.strip() if elem is not None and elem.text else ""


def parse_filename(file_stem: str) -> Optional[dict[str, object]]:
    # Example: zm1_10_img001 -> species=zm, experiment=1, dish=10, frame=001
    if "_img" not in file_stem:
        return None

    left, frame_text = file_stem.rsplit("_img", maxsplit=1)
    if "_" not in left or not frame_text.isdigit():
        return None

    species_and_exp, dish_text = left.split("_", maxsplit=1)
    species_code = "".join(ch for ch in species_and_exp if ch.isalpha()).lower()
    exp_text = species_and_exp[len(species_code):]

    if not species_code or not exp_text.isdigit() or not dish_text.isdigit():
        return None

    experiment_id = int(exp_text)
    dish_id = int(dish_text)
    frame_id = int(frame_text)

    return {
        "species_code_from_filename": species_code,
        "experiment_id": experiment_id,
        "dish_id": dish_id,
        "frame_id": frame_id,
        "sequence_id": f"{species_code}{experiment_id}_{dish_id}",
    }


def find_image(img_dir: Path, file_stem: str) -> Optional[Path]:
    for extension in IMAGE_EXTENSIONS:
        image_path = img_dir / f"{file_stem}{extension}"
        if image_path.exists():
            return image_path
    return None


def read_xml_size(root: ET.Element) -> tuple[int, int, int]:
    size = root.find("size")
    if size is None:
        raise ValueError("missing <size>")

    width = int(float(text_of(size.find("width"))))
    height = int(float(text_of(size.find("height"))))
    depth_text = text_of(size.find("depth"))
    depth = int(float(depth_text)) if depth_text else 3

    return width, height, depth


def read_bbox(obj: ET.Element) -> tuple[int, int, int, int]:
    box = obj.find("bndbox")
    if box is None:
        raise ValueError("missing <bndbox>")

    xmin = round(float(text_of(box.find("xmin"))))
    ymin = round(float(text_of(box.find("ymin"))))
    xmax = round(float(text_of(box.find("xmax"))))
    ymax = round(float(text_of(box.find("ymax"))))

    return xmin, ymin, xmax, ymax


def relative_to_cwd(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()

    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def collect_object_rows(
    raw_root: Path,
    max_xml_per_species: int = 0,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    for species_dir in sorted(raw_root.iterdir()):
        if not species_dir.is_dir() or species_dir.name not in SPECIES_BY_FOLDER:
            continue

        img_dir = species_dir / "img"
        ann_dir = species_dir / "true_ann"

        if not img_dir.exists() or not ann_dir.exists():
            errors.append(
                {
                    "path": relative_to_cwd(species_dir),
                    "error": "missing_img_or_true_ann_dir",
                    "detail": "",
                }
            )
            continue

        species_info = SPECIES_BY_FOLDER[species_dir.name]
        xml_paths = sorted(ann_dir.glob("*.xml"))
        if max_xml_per_species > 0:
            xml_paths = xml_paths[:max_xml_per_species]

        print(f"Scanning {species_dir.name}: {len(xml_paths)} XML files")

        for xml_path in xml_paths:
            file_stem = xml_path.stem
            parsed = parse_filename(file_stem)
            if parsed is None:
                errors.append(
                    {
                        "path": relative_to_cwd(xml_path),
                        "error": "filename_parse_error",
                        "detail": file_stem,
                    }
                )
                continue

            image_path = find_image(img_dir, file_stem)
            if image_path is None:
                errors.append(
                    {
                        "path": relative_to_cwd(xml_path),
                        "error": "missing_image",
                        "detail": file_stem,
                    }
                )
                continue

            try:
                root = ET.parse(xml_path).getroot()
                image_width, image_height, image_depth = read_xml_size(root)
            except Exception as exc:
                errors.append(
                    {
                        "path": relative_to_cwd(xml_path),
                        "error": "xml_read_error",
                        "detail": str(exc),
                    }
                )
                continue

            for object_id, obj in enumerate(root.findall("object")):
                raw_label = text_of(obj.find("name"))
                if raw_label not in LABEL_MAP:
                    errors.append(
                        {
                            "path": relative_to_cwd(xml_path),
                            "error": "unknown_label",
                            "detail": f"object_id={object_id}, label={raw_label}",
                        }
                    )
                    continue

                try:
                    xmin, ymin, xmax, ymax = read_bbox(obj)
                except Exception as exc:
                    errors.append(
                        {
                            "path": relative_to_cwd(xml_path),
                            "error": "bbox_read_error",
                            "detail": f"object_id={object_id}, {exc}",
                        }
                    )
                    continue

                if xmin >= xmax or ymin >= ymax:
                    errors.append(
                        {
                            "path": relative_to_cwd(xml_path),
                            "error": "invalid_bbox",
                            "detail": f"object_id={object_id}, {(xmin, ymin, xmax, ymax)}",
                        }
                    )
                    continue

                if xmin < 0 or ymin < 0 or xmax > image_width or ymax > image_height:
                    errors.append(
                        {
                            "path": relative_to_cwd(xml_path),
                            "error": "bbox_out_of_bounds",
                            "detail": f"object_id={object_id}, {(xmin, ymin, xmax, ymax)}",
                        }
                    )
                    continue

                label, class_id = LABEL_MAP[raw_label]
                crop_filename = f"{file_stem}_obj{object_id:03d}.jpg"

                rows.append(
                    {
                        "image_path": relative_to_cwd(image_path),
                        "xml_path": relative_to_cwd(xml_path),
                        "filename": image_path.name,
                        "file_stem": file_stem,
                        "folder_name": species_dir.name,
                        "species_code": species_info["species_code"],
                        "species_name": species_info["species_name"],
                        "vietnamese_name": species_info["vietnamese_name"],
                        "species_code_from_filename": parsed["species_code_from_filename"],
                        "experiment_id": parsed["experiment_id"],
                        "dish_id": parsed["dish_id"],
                        "frame_id": parsed["frame_id"],
                        "sequence_id": parsed["sequence_id"],
                        "image_width": image_width,
                        "image_height": image_height,
                        "image_depth": image_depth,
                        "object_id": object_id,
                        "raw_label": raw_label,
                        "label": label,
                        "class_id": class_id,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "bbox_width": xmax - xmin,
                        "bbox_height": ymax - ymin,
                        "bbox_area": (xmax - xmin) * (ymax - ymin),
                        "crop_filename": crop_filename,
                    }
                )

    return rows, errors


def ensure_empty_crop_dir(crops_dir: Path, overwrite: bool) -> None:
    existing_files = [
        path for path in crops_dir.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    ] if crops_dir.exists() else []

    if existing_files and not overwrite:
        raise RuntimeError(
            f"{crops_dir} already contains crop files. "
            "Use --overwrite to recreate it."
        )

    if overwrite and existing_files:
        for path in existing_files:
            path.unlink()

    for split in SPLITS:
        for label in ("germinated", "non_germinated"):
            (crops_dir / split / label).mkdir(parents=True, exist_ok=True)


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def crop_objects(
    rows: list[dict[str, object]],
    crops_dir: Path,
    padding: int = 0,
    resize: Optional[int] = None,
) -> list[dict[str, object]]:
    rows_by_image: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_image[str(row["image_path"])].append(row)

    updated_rows: list[dict[str, object]] = []
    saved = 0

    for image_path_text, image_rows in rows_by_image.items():
        image_path = resolve_project_path(image_path_text)

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size

            for row in image_rows:
                xmin = max(0, int(row["xmin"]) - padding)
                ymin = max(0, int(row["ymin"]) - padding)
                xmax = min(width, int(row["xmax"]) + padding)
                ymax = min(height, int(row["ymax"]) + padding)

                crop = image.crop((xmin, ymin, xmax, ymax))
                if resize is not None:
                    crop = crop.resize((resize, resize), Image.Resampling.BILINEAR)

                split = str(row["split"])
                label = str(row["label"])
                crop_path = crops_dir / split / label / str(row["crop_filename"])
                crop.save(crop_path, quality=95)

                updated_row = dict(row)
                updated_row["crop_path"] = relative_to_cwd(crop_path)
                updated_row["crop_width"] = crop.width
                updated_row["crop_height"] = crop.height
                updated_rows.append(updated_row)
                saved += 1

        if saved and saved % 25000 == 0:
            print(f"  saved crops: {saved}")

    print(f"Saved crops: {saved}")
    return updated_rows


def write_error_rows(path: Path, rows: Iterable[dict[str, str]]) -> None:
    write_rows(path, rows, ["path", "error", "detail"])


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_split_metadata(metadata_dir: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    write_rows(metadata_dir / "all_objects_with_split.csv", rows, fieldnames)
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        write_rows(metadata_dir / f"{split}.csv", split_rows, fieldnames)


def print_object_summary(rows: list[dict[str, object]], errors: list[dict[str, str]]) -> None:
    species_counts = Counter(str(row["species_code"]) for row in rows)
    label_counts = Counter(str(row["label"]) for row in rows)

    print("\nObject metadata summary:")
    print(f"  objects: {len(rows)}")
    print(f"  sequences: {len({row['sequence_id'] for row in rows})}")
    print(f"  images: {len({row['filename'] for row in rows})}")
    print(f"  skipped/errors: {len(errors)}")

    print("\nObjects by species:")
    for species, count in sorted(species_counts.items()):
        print(f"  {species}: {count}")

    print("\nObjects by label:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create crop-classification dataset for the Custom CNN baseline."
    )
    parser.add_argument("--raw-root", default="data/raw/GermPredDataset")
    parser.add_argument("--metadata-dir", default="data/metadata")
    parser.add_argument("--crops-dir", default="data/crops")
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument(
        "--resize",
        type=int,
        default=0,
        help="Optionally save square resized crops. Default keeps original crop sizes.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Create CSV metadata and split files without writing crop images.",
    )
    parser.add_argument(
        "--from-metadata",
        action="store_true",
        help="Create crop images from data/metadata/all_objects_with_split.csv.",
    )
    parser.add_argument(
        "--max-xml-per-species",
        type=int,
        default=0,
        help="Debug option for quick dry runs. Default processes every XML file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated crop images under data/crops.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    metadata_dir = Path(args.metadata_dir)
    crops_dir = Path(args.crops_dir)

    if not raw_root.exists():
        print(f"Raw dataset not found: {raw_root}")
        return 1

    metadata_dir.mkdir(parents=True, exist_ok=True)

    if args.from_metadata:
        split_metadata_path = metadata_dir / "all_objects_with_split.csv"
        if not split_metadata_path.exists():
            print(f"Split metadata not found: {split_metadata_path}")
            return 1

        rows_with_split = read_rows(split_metadata_path)
        if not rows_with_split:
            print(f"Split metadata is empty: {split_metadata_path}")
            return 1

        if args.metadata_only:
            print_summary(rows_with_split)
            print("\nMetadata-only mode: crop images were not written.")
            return 0

        ensure_empty_crop_dir(crops_dir, overwrite=args.overwrite)
        resize = args.resize if args.resize > 0 else None
        rows_with_crops = crop_objects(
            rows=rows_with_split,
            crops_dir=crops_dir,
            padding=max(args.padding, 0),
            resize=resize,
        )
        write_split_metadata(metadata_dir, rows_with_crops)
        print_summary(rows_with_crops)

        print("\nSaved files:")
        print(f"  {metadata_dir / 'all_objects_with_split.csv'}")
        for split in SPLITS:
            print(f"  {metadata_dir / f'{split}.csv'}")
        print(f"  {crops_dir}")
        return 0

    object_rows, errors = collect_object_rows(
        raw_root=raw_root,
        max_xml_per_species=max(args.max_xml_per_species, 0),
    )
    if not object_rows:
        print("No objects found. Crop dataset was not created.")
        write_error_rows(metadata_dir / "all_objects_errors.csv", errors)
        return 1

    all_objects_fieldnames = list(object_rows[0].keys())
    write_rows(metadata_dir / "all_objects.csv", object_rows, all_objects_fieldnames)
    write_error_rows(metadata_dir / "all_objects_errors.csv", errors)
    print_object_summary(object_rows, errors)

    sequence_rows = assign_splits(make_sequence_rows(object_rows))
    rows_with_split = attach_split(object_rows, sequence_rows)

    if not check_leakage(rows_with_split):
        return 1

    sequence_fieldnames = [
        "sequence_id",
        "species_code",
        "species_name",
        "num_images",
        "num_objects",
        "germinated_count",
        "non_germinated_count",
        "germinated_ratio",
        "split",
    ]
    write_rows(metadata_dir / "sequence_split.csv", sequence_rows, sequence_fieldnames)

    if args.metadata_only:
        write_split_metadata(metadata_dir, rows_with_split)
        print_summary(rows_with_split)
        print("\nMetadata-only mode: crop images were not written.")
        return 0

    ensure_empty_crop_dir(crops_dir, overwrite=args.overwrite)
    resize = args.resize if args.resize > 0 else None
    rows_with_crops = crop_objects(
        rows=rows_with_split,
        crops_dir=crops_dir,
        padding=max(args.padding, 0),
        resize=resize,
    )

    write_split_metadata(metadata_dir, rows_with_crops)
    print_summary(rows_with_crops)

    print("\nSaved files:")
    print(f"  {metadata_dir / 'all_objects.csv'}")
    print(f"  {metadata_dir / 'all_objects_with_split.csv'}")
    print(f"  {metadata_dir / 'sequence_split.csv'}")
    for split in SPLITS:
        print(f"  {metadata_dir / f'{split}.csv'}")
    print(f"  {metadata_dir / 'all_objects_errors.csv'}")
    print(f"  {crops_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
