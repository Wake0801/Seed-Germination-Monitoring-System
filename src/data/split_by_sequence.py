#!/usr/bin/env python3
"""Split object metadata by sequence_id for crop classification.

The same petri dish appears across many time-lapse frames, so every object
from one sequence_id must stay in exactly one split.

Usage:
  python src/data/split_by_sequence.py
  python src/data/split_by_sequence.py --metadata-dir data/metadata
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


SPLITS = ("train", "val", "test")
RANDOM_SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def required_columns_present(rows: list[dict[str, str]], required: set[str]) -> bool:
    if not rows:
        print("Input metadata is empty.")
        return False

    missing = required - set(rows[0].keys())
    if missing:
        print("Missing required columns:")
        for column in sorted(missing):
            print(f"  - {column}")
        return False

    return True


def make_sequence_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["sequence_id"]].append(row)

    sequence_rows = []

    for sequence_id, group in grouped.items():
        label_counts = Counter(row["label"] for row in group)
        filenames = {row["filename"] for row in group}
        first = group[0]

        sequence_rows.append(
            {
                "sequence_id": sequence_id,
                "species_code": first["species_code"],
                "species_name": first["species_name"],
                "num_images": len(filenames),
                "num_objects": len(group),
                "germinated_count": label_counts.get("germinated", 0),
                "non_germinated_count": label_counts.get("non_germinated", 0),
            }
        )

    for row in sequence_rows:
        total = int(row["num_objects"])
        row["germinated_ratio"] = (
            float(row["germinated_count"]) / total if total else 0.0
        )

    return sequence_rows


def assign_splits(sequence_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rng = random.Random(RANDOM_SEED)
    by_species: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in sequence_rows:
        by_species[str(row["species_code"])].append(row)

    split_rows: list[dict[str, object]] = []

    print("Splitting sequences by species:")
    for species_code in sorted(by_species):
        species_rows = by_species[species_code]
        rng.shuffle(species_rows)

        n = len(species_rows)
        n_train = round(n * TRAIN_RATIO)
        n_val = round(n * VAL_RATIO)

        for index, row in enumerate(species_rows):
            if index < n_train:
                split = "train"
            elif index < n_train + n_val:
                split = "val"
            else:
                split = "test"

            row = dict(row)
            row["split"] = split
            split_rows.append(row)

        print(
            f"  {species_code}: total={n}, "
            f"train={n_train}, val={n_val}, test={n - n_train - n_val}"
        )

    return split_rows


def attach_split(
    object_rows: list[dict[str, str]],
    sequence_rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    split_by_sequence = {
        str(row["sequence_id"]): str(row["split"])
        for row in sequence_rows
    }

    output_rows = []
    for row in object_rows:
        sequence_id = row["sequence_id"]
        if sequence_id not in split_by_sequence:
            raise KeyError(f"Missing split for sequence_id={sequence_id}")

        output_row = dict(row)
        output_row["split"] = split_by_sequence[sequence_id]
        output_rows.append(output_row)

    return output_rows


def check_leakage(rows: list[dict[str, str]]) -> bool:
    splits_by_sequence: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_sequence[row["sequence_id"]].add(row["split"])

    leaked = {
        sequence_id: splits
        for sequence_id, splits in splits_by_sequence.items()
        if len(splits) > 1
    }

    if leaked:
        print("Data leakage detected:")
        for sequence_id, splits in list(leaked.items())[:20]:
            print(f"  {sequence_id}: {sorted(splits)}")
        return False

    print("Leakage check: PASSED")
    return True


def print_summary(rows: list[dict[str, str]]) -> None:
    split_counts = Counter(row["split"] for row in rows)
    label_counts = Counter((row["split"], row["label"]) for row in rows)
    species_counts = Counter((row["split"], row["species_code"]) for row in rows)

    print("\nObject rows by split:")
    for split in SPLITS:
        print(f"  {split}: {split_counts.get(split, 0)}")

    print("\nObject rows by split and label:")
    for split in SPLITS:
        germinated = label_counts.get((split, "germinated"), 0)
        non_germinated = label_counts.get((split, "non_germinated"), 0)
        print(
            f"  {split}: germinated={germinated}, "
            f"non_germinated={non_germinated}"
        )

    print("\nObject rows by split and species:")
    species_codes = sorted({row["species_code"] for row in rows})
    for split in SPLITS:
        values = ", ".join(
            f"{species}={species_counts.get((split, species), 0)}"
            for species in species_codes
        )
        print(f"  {split}: {values}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split all_objects.csv by sequence_id."
    )
    parser.add_argument("--metadata-dir", default="data/metadata")
    parser.add_argument("--input", default="all_objects.csv")
    parser.add_argument("--sequence-output", default="sequence_split.csv")
    parser.add_argument("--object-output", default="all_objects_with_split.csv")
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir)
    input_path = metadata_dir / args.input

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    object_rows = read_rows(input_path)
    required = {
        "filename",
        "species_code",
        "species_name",
        "sequence_id",
        "label",
    }

    if not required_columns_present(object_rows, required):
        return 1

    sequence_rows = make_sequence_rows(object_rows)
    sequence_rows = assign_splits(sequence_rows)
    object_rows_with_split = attach_split(object_rows, sequence_rows)

    if not check_leakage(object_rows_with_split):
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

    object_fieldnames = list(object_rows_with_split[0].keys())

    write_rows(metadata_dir / args.sequence_output, sequence_rows, sequence_fieldnames)
    write_rows(metadata_dir / args.object_output, object_rows_with_split, object_fieldnames)

    for split in SPLITS:
        split_rows = [row for row in object_rows_with_split if row["split"] == split]
        write_rows(metadata_dir / f"{split}.csv", split_rows, object_fieldnames)

    print_summary(object_rows_with_split)

    print("\nSaved files:")
    print(f"  {metadata_dir / args.sequence_output}")
    print(f"  {metadata_dir / args.object_output}")
    for split in SPLITS:
        print(f"  {metadata_dir / f'{split}.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
