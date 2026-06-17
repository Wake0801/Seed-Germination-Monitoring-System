#!/usr/bin/env python3
"""Validate Pascal VOC image/annotation integrity across a dataset tree.

Usage:
  python validate_pascal_voc.py /path/to/dataset/root
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


EXPECTED_SIZE = (624, 624)

EXPECTED_LABELS = {
    "zm_im", "zm_el",
    "sc_im", "sc_el",
    "pg_im", "pg_el",
}


def find_subdir(root: Path, name: str) -> Optional[Path]:
    candidate = root / name
    return candidate if candidate.exists() and candidate.is_dir() else None


def collect_files(folder: Path, extensions: Tuple[str, ...]) -> Dict[str, Path]:
    files: Dict[str, Path] = {}

    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            files[path.stem] = path

    return files


def parse_xml(xml_path: Path) -> Tuple[Optional[ET.ElementTree], Optional[str]]:
    try:
        tree = ET.parse(xml_path)
        return tree, None
    except ET.ParseError as exc:
        return None, str(exc)


def text_of(elem: Optional[ET.Element]) -> str:
    return elem.text.strip() if elem is not None and elem.text else ""


def validate_xml(
    xml_path: Path,
    image_files: Dict[str, Path],
    verbose: bool = False,
) -> List[str]:
    errors: List[str] = []

    tree, parse_error = parse_xml(xml_path)

    if parse_error:
        errors.append(f"Invalid XML: {parse_error}")
        return errors

    root = tree.getroot()

    # =========================
    # 1. Kiểm tra filename
    # =========================

    filename_node = root.find("filename")
    filename = text_of(filename_node)

    image_path = None

    if not filename:
        errors.append("Missing <filename>")

        # Nếu XML thiếu filename thì thử tìm ảnh theo tên file XML
        if xml_path.stem in image_files:
            image_path = image_files[xml_path.stem]
    else:
        image_stem = Path(filename).stem

        if image_stem not in image_files:
            errors.append(f"Referenced image missing: {filename}")
        else:
            image_path = image_files[image_stem]

    # =========================
    # 2. Kiểm tra ảnh thật
    # =========================

    img_width = img_height = None

    if image_path is not None:
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as exc:
            errors.append(f"Image read error: {exc}")

        if img_width is not None and img_height is not None:
            if (img_width, img_height) != EXPECTED_SIZE:
                errors.append(
                    f"Wrong image size: expected {EXPECTED_SIZE}, got {(img_width, img_height)}"
                )

    # =========================
    # 3. Kiểm tra size trong XML
    # =========================

    size_node = root.find("size")
    width = height = depth = None

    if size_node is None:
        errors.append("Missing <size>")
    else:
        width_node = size_node.find("width")
        height_node = size_node.find("height")
        depth_node = size_node.find("depth")

        if width_node is not None:
            try:
                width = int(float(text_of(width_node)))
            except ValueError:
                errors.append(f"Invalid width value: {text_of(width_node)}")
        else:
            errors.append("Missing width value")

        if height_node is not None:
            try:
                height = int(float(text_of(height_node)))
            except ValueError:
                errors.append(f"Invalid height value: {text_of(height_node)}")
        else:
            errors.append("Missing height value")

        if depth_node is not None:
            try:
                depth = int(float(text_of(depth_node)))
            except ValueError:
                errors.append(f"Invalid depth value: {text_of(depth_node)}")

    # Nếu đọc được ảnh thật thì so sánh size XML với ảnh
    if (
        width is not None
        and height is not None
        and img_width is not None
        and img_height is not None
    ):
        if width != img_width or height != img_height:
            errors.append(
                f"XML size mismatch: XML {(width, height)} != image {(img_width, img_height)}"
            )

    # Nếu XML thiếu width/height thì dùng size ảnh thật để kiểm tra bbox
    if width is None and img_width is not None:
        width = img_width

    if height is None and img_height is not None:
        height = img_height

    # =========================
    # 4. Kiểm tra object và bbox
    # =========================

    objects = root.findall("object")

    if not objects:
        errors.append("XML has no object")

    for obj in objects:
        name_node = obj.find("name")
        label = text_of(name_node) or "<unnamed>"

        if label == "<unnamed>":
            errors.append("Object missing label")
        elif label not in EXPECTED_LABELS:
            errors.append(f"Unexpected label: {label}")

        bndbox = obj.find("bndbox")

        if bndbox is None:
            errors.append(f"Object '{label}' missing <bndbox>")
            continue

        coords = {}

        for coord in ("xmin", "ymin", "xmax", "ymax"):
            node = bndbox.find(coord)

            if node is None or not text_of(node):
                errors.append(f"Object '{label}' missing {coord}")
                continue

            try:
                value = text_of(node)

                # Sửa quan trọng:
                # chấp nhận cả tọa độ dạng int như 123
                # và dạng float như 123.0 hoặc 123.5
                coords[coord] = round(float(value))

            except ValueError:
                errors.append(f"Object '{label}' invalid {coord} value: {text_of(node)}")

        if set(coords) == {"xmin", "ymin", "xmax", "ymax"}:
            xmin = coords["xmin"]
            ymin = coords["ymin"]
            xmax = coords["xmax"]
            ymax = coords["ymax"]

            if xmin >= xmax:
                errors.append(f"Object '{label}' xmin >= xmax")

            if ymin >= ymax:
                errors.append(f"Object '{label}' ymin >= ymax")

            if width is not None and height is not None:
                if not (0 <= xmin < width):
                    errors.append(f"Object '{label}' xmin out of bounds: {xmin}")

                if not (0 <= xmax <= width):
                    errors.append(f"Object '{label}' xmax out of bounds: {xmax}")

                if not (0 <= ymin < height):
                    errors.append(f"Object '{label}' ymin out of bounds: {ymin}")

                if not (0 <= ymax <= height):
                    errors.append(f"Object '{label}' ymax out of bounds: {ymax}")

    if verbose and not errors:
        errors.append("OK")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Pascal VOC image/annotation pairs."
    )

    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Dataset root folder containing class subfolders",
    )

    parser.add_argument(
        "--extensions",
        default=".jpg,.jpeg,.png,.bmp,.tif,.tiff",
        help="Comma-separated image extensions",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-annotation status",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    if not root.exists() or not root.is_dir():
        print(f"Error: root folder not found: {root}")
        return 2

    image_extensions = tuple(
        ext.strip().lower()
        for ext in args.extensions.split(",")
        if ext.strip()
    )

    folders = [p for p in root.iterdir() if p.is_dir()]

    if not folders:
        print(f"No dataset subfolders found under {root}")
        return 1

    summary = []
    total_xml_errors = 0
    total_missing_images = 0
    total_missing_annotations = 0

    error_counter = Counter()

    for plant_folder in sorted(folders):
        img_folder = find_subdir(plant_folder, "img")
        ann_folder = find_subdir(plant_folder, "true_ann")

        print(f"\nScanning {plant_folder.name}")

        if img_folder is None or ann_folder is None:
            print(f"  Missing expected subfolder in {plant_folder}")

            if img_folder is None:
                print("   - missing img/")

            if ann_folder is None:
                print("   - missing true_ann/")

            total_xml_errors += 1
            continue

        images = collect_files(img_folder, image_extensions)
        xmls = collect_files(ann_folder, (".xml",))

        missing_images = sorted(name for name in xmls if name not in images)
        missing_annotations = sorted(name for name in images if name not in xmls)

        total_missing_images += len(missing_images)
        total_missing_annotations += len(missing_annotations)

        if missing_images:
            print(f"  {len(missing_images)} XMLs reference missing images")
            for name in missing_images[:20]:
                print(f"   - {name}")

        if missing_annotations:
            print(f"  {len(missing_annotations)} images missing annotations")
            for name in missing_annotations[:20]:
                print(f"   - {name}")

        if not missing_images and not missing_annotations:
            print(f"  All image/XML base names match ({len(images)} pairs)")

        plant_errors = 0

        for xml_name, xml_path in sorted(xmls.items()):
            errors = validate_xml(xml_path, images, verbose=args.verbose)

            if errors and not (args.verbose and errors == ["OK"]):
                plant_errors += 1
                total_xml_errors += 1

                print(f"  XML {xml_path.name} -> {len(errors)} issue(s)")

                for err in errors:
                    print(f"    - {err}")
                    error_counter[err] += 1

            elif args.verbose:
                print(f"  XML {xml_path.name} -> OK")

        summary.append(
            (
                plant_folder.name,
                len(images),
                len(xmls),
                len(missing_images),
                len(missing_annotations),
                plant_errors,
            )
        )

    print("\nSummary:")

    for (
        plant_name,
        img_count,
        xml_count,
        missing_imgs,
        missing_anns,
        plant_errors,
    ) in summary:
        print(
            f"  {plant_name}: "
            f"images={img_count}, "
            f"xmls={xml_count}, "
            f"missing_image_refs={missing_imgs}, "
            f"missing_annotations={missing_anns}, "
            f"xml_errors={plant_errors}"
        )

    print(f"\nTotal missing image refs: {total_missing_images}")
    print(f"Total missing annotations: {total_missing_annotations}")
    print(f"Total XML files with issues: {total_xml_errors}")

    print("\nError type summary:")

    if error_counter:
        for err, count in error_counter.most_common():
            print(f"  {err}: {count}")
    else:
        print("  No XML/content errors found.")

    total_issues = (
        total_missing_images
        + total_missing_annotations
        + total_xml_errors
    )

    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())