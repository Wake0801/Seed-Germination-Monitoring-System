#!/usr/bin/env python3
"""Split dataset into train/val/test by sequence_id.

Each sequence_id represents one petri dish over time.
All images/objects from the same sequence_id are assigned to the same split.

Usage:
  python split_by_sequence.py .
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


def assign_split_for_species(sequence_df: pd.DataFrame) -> pd.DataFrame:
    """
    Split sequences inside one species into train/val/test.

    We split by sequence_id, not by images or objects.
    """

    sequence_df = sequence_df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    n = len(sequence_df)

    n_train = round(n * TRAIN_RATIO)
    n_val = round(n * VAL_RATIO)

    # Phần còn lại là test
    n_test = n - n_train - n_val

    sequence_df["split"] = "test"

    sequence_df.loc[: n_train - 1, "split"] = "train"
    sequence_df.loc[n_train : n_train + n_val - 1, "split"] = "val"
    sequence_df.loc[n_train + n_val :, "split"] = "test"

    print(
        f"  {sequence_df['species_code'].iloc[0]}: "
        f"total={n}, train={n_train}, val={n_val}, test={n_test}"
    )

    return sequence_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split object metadata into train/val/test by sequence_id."
    )

    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Dataset root folder containing object_metadata.csv",
    )

    parser.add_argument(
        "--input",
        default="object_metadata.csv",
        help="Input object-level metadata CSV file.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    input_path = root / args.input

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    df = pd.read_csv(input_path)

    required_columns = {
        "image_path",
        "xml_path",
        "filename",
        "species_code",
        "species_name",
        "sequence_id",
        "binary_label",
        "raw_label",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        print("Missing required columns:")
        for col in sorted(missing_columns):
            print(f"  - {col}")
        return 1

    print("Loaded object metadata:")
    print(f"  Total object rows: {len(df)}")
    print(f"  Total images: {df['filename'].nunique()}")
    print(f"  Total sequences: {df['sequence_id'].nunique()}")

    # =========================
    # 1. Tạo bảng sequence-level
    # =========================

    sequence_df = (
        df.groupby(["sequence_id", "species_code", "species_name"])
        .agg(
            num_images=("filename", "nunique"),
            num_objects=("filename", "size"),
            germinated_count=(
                "binary_label",
                lambda x: (x == "germinated").sum(),
            ),
            non_germinated_count=(
                "binary_label",
                lambda x: (x == "non-germinated").sum(),
            ),
        )
        .reset_index()
    )

    sequence_df["germinated_ratio"] = (
        sequence_df["germinated_count"] / sequence_df["num_objects"]
    )

    print("\nSequence summary before split:")
    print(f"  Total sequences: {len(sequence_df)}")

    print("\nSequences by species:")
    print(sequence_df["species_code"].value_counts().sort_index())

    # =========================
    # 2. Chia split theo từng loài
    # =========================

    print("\nSplitting sequences by species:")

    split_parts = []

    for species_code, species_seq_df in sequence_df.groupby("species_code"):
        split_part = assign_split_for_species(species_seq_df)
        split_parts.append(split_part)

    split_sequence_df = pd.concat(split_parts, ignore_index=True)

    # =========================
    # 3. Gán split ngược lại cho từng object
    # =========================

    split_map = split_sequence_df[["sequence_id", "split"]]

    df_with_split = df.merge(
        split_map,
        on="sequence_id",
        how="left",
        validate="many_to_one",
    )

    if df_with_split["split"].isna().any():
        print("Error: some rows do not have split assigned.")
        return 1

    # =========================
    # 4. Tạo bảng image-level split
    # =========================

    image_split_df = (
        df_with_split[
            [
                "image_path",
                "xml_path",
                "filename",
                "species_code",
                "species_name",
                "experiment_id",
                "dish_id",
                "frame_id",
                "sequence_id",
                "split",
            ]
        ]
        .drop_duplicates()
        .sort_values(["split", "species_code", "sequence_id", "frame_id"])
        .reset_index(drop=True)
    )

    # =========================
    # 5. Kiểm tra data leakage
    # =========================

    leakage_check = (
        image_split_df.groupby("sequence_id")["split"]
        .nunique()
        .reset_index(name="num_splits")
    )

    leaked_sequences = leakage_check[leakage_check["num_splits"] > 1]

    if len(leaked_sequences) > 0:
        print("Data leakage detected! Some sequence_id appears in multiple splits.")
        print(leaked_sequences.head(20))
        return 1

    print("\nLeakage check: PASSED")
    print("  Each sequence_id appears in exactly one split.")

    # =========================
    # 6. In thống kê split
    # =========================

    print("\n===== SPLIT SUMMARY BY SEQUENCE =====")
    print(
        split_sequence_df.groupby(["split", "species_code"])
        .size()
        .unstack(fill_value=0)
    )

    print("\n===== SPLIT SUMMARY BY IMAGE =====")
    print(
        image_split_df.groupby(["split", "species_code"])
        .size()
        .unstack(fill_value=0)
    )

    print("\n===== SPLIT SUMMARY BY OBJECT =====")
    print(
        df_with_split.groupby(["split", "species_code"])
        .size()
        .unstack(fill_value=0)
    )

    print("\n===== BINARY LABEL DISTRIBUTION BY SPLIT =====")
    print(
        df_with_split.groupby(["split", "binary_label"])
        .size()
        .unstack(fill_value=0)
    )

    print("\n===== RAW LABEL DISTRIBUTION BY SPLIT =====")
    print(
        df_with_split.groupby(["split", "raw_label"])
        .size()
        .unstack(fill_value=0)
    )

    # =========================
    # 7. Lưu file output
    # =========================

    sequence_split_path = root / "sequence_split.csv"
    image_split_path = root / "image_split.csv"
    object_split_path = root / "object_metadata_with_split.csv"

    split_sequence_df = split_sequence_df.sort_values(
        ["split", "species_code", "sequence_id"]
    ).reset_index(drop=True)

    df_with_split = df_with_split.sort_values(
        ["split", "species_code", "sequence_id", "frame_id", "object_id"]
    ).reset_index(drop=True)

    split_sequence_df.to_csv(sequence_split_path, index=False, encoding="utf-8")
    image_split_df.to_csv(image_split_path, index=False, encoding="utf-8")
    df_with_split.to_csv(object_split_path, index=False, encoding="utf-8")

    print("\nSaved files:")
    print(f"  {sequence_split_path}")
    print(f"  {image_split_path}")
    print(f"  {object_split_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())