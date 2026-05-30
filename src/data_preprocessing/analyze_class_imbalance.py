#!/usr/bin/env python3
"""Analyze class imbalance after train/val/test split.

Input:
  object_metadata_with_split.csv

Outputs:
  imbalance_overall_binary.csv
  imbalance_by_split_binary.csv
  imbalance_by_species_binary.csv
  imbalance_raw_label_by_split.csv
  class_weights_binary_train.json
  imbalance_report.txt

Usage:
  python analyze_class_imbalance.py .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def make_distribution(
    df: pd.DataFrame,
    group_cols: list[str],
    label_col: str,
) -> pd.DataFrame:
    counts = (
        df.groupby(group_cols + [label_col])
        .size()
        .reset_index(name="count")
    )

    totals = (
        df.groupby(group_cols)
        .size()
        .reset_index(name="total")
    )

    result = counts.merge(totals, on=group_cols, how="left")
    result["percent"] = result["count"] / result["total"] * 100

    return result.sort_values(group_cols + [label_col]).reset_index(drop=True)


def compute_class_weights_from_train(
    train_df: pd.DataFrame,
    label_col: str = "binary_label",
) -> dict[str, float]:
    """Compute balanced class weights using sklearn-style formula.

    weight_i = total_samples / (num_classes * count_i)

    These weights are for reference/reporting.
    """

    counts = train_df[label_col].value_counts().to_dict()

    total = sum(counts.values())
    num_classes = len(counts)

    weights = {
        label: total / (num_classes * count)
        for label, count in counts.items()
    }

    return dict(sorted(weights.items()))


def imbalance_ratio(counts: dict[str, int]) -> float:
    if not counts:
        return 0.0

    min_count = min(counts.values())
    max_count = max(counts.values())

    if min_count == 0:
        return float("inf")

    return max_count / min_count


def severity_from_ratio(ratio: float) -> str:
    if ratio < 1.5:
        return "mild"
    if ratio < 3.0:
        return "moderate"
    return "severe"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze class imbalance from object metadata with split."
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

    args = parser.parse_args()

    root = Path(args.root).resolve()
    input_path = root / args.input

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    df = pd.read_csv(input_path)

    required_columns = {
        "split",
        "species_code",
        "species_name",
        "sequence_id",
        "filename",
        "raw_label",
        "binary_label",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        print("Missing required columns:")
        for col in sorted(missing_columns):
            print(f"  - {col}")
        return 1

    print("Loaded metadata:")
    print(f"  Total objects: {len(df)}")
    print(f"  Total images: {df['filename'].nunique()}")
    print(f"  Total sequences: {df['sequence_id'].nunique()}")

    # =========================
    # 1. Overall binary distribution
    # =========================

    overall_binary = (
        df["binary_label"]
        .value_counts()
        .rename_axis("binary_label")
        .reset_index(name="count")
    )

    overall_binary["total"] = len(df)
    overall_binary["percent"] = overall_binary["count"] / len(df) * 100
    overall_binary = overall_binary.sort_values("binary_label").reset_index(drop=True)

    # =========================
    # 2. Binary distribution by split
    # =========================

    split_binary = make_distribution(
        df=df,
        group_cols=["split"],
        label_col="binary_label",
    )

    # =========================
    # 3. Binary distribution by species
    # =========================

    species_binary = make_distribution(
        df=df,
        group_cols=["species_code", "species_name"],
        label_col="binary_label",
    )

    # =========================
    # 4. Raw label distribution by split
    # =========================

    raw_by_split = make_distribution(
        df=df,
        group_cols=["split"],
        label_col="raw_label",
    )

    # =========================
    # 5. Train class weights
    # =========================

    train_df = df[df["split"] == "train"].copy()

    train_counts = train_df["binary_label"].value_counts().to_dict()
    train_weights = compute_class_weights_from_train(
        train_df=train_df,
        label_col="binary_label",
    )

    overall_counts = df["binary_label"].value_counts().to_dict()
    overall_ratio = imbalance_ratio(overall_counts)
    train_ratio = imbalance_ratio(train_counts)

    # =========================
    # 6. Save files
    # =========================

    overall_path = root / "imbalance_overall_binary.csv"
    split_path = root / "imbalance_by_split_binary.csv"
    species_path = root / "imbalance_by_species_binary.csv"
    raw_split_path = root / "imbalance_raw_label_by_split.csv"
    weights_path = root / "class_weights_binary_train.json"
    report_path = root / "imbalance_report.txt"

    overall_binary.to_csv(overall_path, index=False, encoding="utf-8")
    split_binary.to_csv(split_path, index=False, encoding="utf-8")
    species_binary.to_csv(species_path, index=False, encoding="utf-8")
    raw_by_split.to_csv(raw_split_path, index=False, encoding="utf-8")

    with weights_path.open("w", encoding="utf-8") as f:
        json.dump(train_weights, f, indent=2, ensure_ascii=False)

    # =========================
    # 7. Print summary
    # =========================

    print("\n===== OVERALL BINARY DISTRIBUTION =====")
    print(overall_binary)

    print("\n===== BINARY DISTRIBUTION BY SPLIT =====")
    print(split_binary)

    print("\n===== BINARY DISTRIBUTION BY SPECIES =====")
    print(species_binary)

    print("\n===== RAW LABEL DISTRIBUTION BY SPLIT =====")
    print(raw_by_split)

    print("\n===== TRAIN CLASS WEIGHTS =====")
    for label, weight in train_weights.items():
        print(f"  {label}: {weight:.4f}")

    print("\n===== IMBALANCE RATIO =====")
    print(f"  Overall binary imbalance ratio: {overall_ratio:.4f}")
    print(f"  Train binary imbalance ratio: {train_ratio:.4f}")
    print(f"  Overall severity: {severity_from_ratio(overall_ratio)}")
    print(f"  Train severity: {severity_from_ratio(train_ratio)}")

    # =========================
    # 8. Write report text
    # =========================

    with report_path.open("w", encoding="utf-8") as f:
        f.write("CLASS IMBALANCE REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("Dataset summary\n")
        f.write(f"- Total objects: {len(df)}\n")
        f.write(f"- Total images: {df['filename'].nunique()}\n")
        f.write(f"- Total sequences: {df['sequence_id'].nunique()}\n\n")

        f.write("Overall binary distribution\n")
        for _, row in overall_binary.iterrows():
            f.write(
                f"- {row['binary_label']}: "
                f"{int(row['count'])} objects "
                f"({row['percent']:.2f}%)\n"
            )

        f.write("\nImbalance level\n")
        f.write(f"- Overall imbalance ratio: {overall_ratio:.4f}\n")
        f.write(f"- Train imbalance ratio: {train_ratio:.4f}\n")
        f.write(f"- Severity: {severity_from_ratio(train_ratio)}\n\n")

        f.write("Train class weights, reference only\n")
        for label, weight in train_weights.items():
            f.write(f"- {label}: {weight:.4f}\n")

        f.write("\nDecision\n")
        if train_ratio < 1.5:
            f.write(
                "The binary class imbalance is mild. "
                "No aggressive resampling is applied at this stage. "
                "The training set is kept unchanged to preserve the natural "
                "time-series and petri dish structure. "
                "Class-level Precision, Recall, F1-score, and mAP should be monitored "
                "during model evaluation. If the germinated class shows low recall, "
                "class weights, focal loss, or train-only weighted sampling can be "
                "considered in later experiments.\n"
            )
        else:
            f.write(
                "The class imbalance is non-trivial. "
                "Additional techniques such as class weights, focal loss, "
                "or train-only weighted sampling should be considered. "
                "Validation and test sets must not be resampled.\n"
            )

    print("\nSaved files:")
    print(f"  {overall_path}")
    print(f"  {split_path}")
    print(f"  {species_path}")
    print(f"  {raw_split_path}")
    print(f"  {weights_path}")
    print(f"  {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())