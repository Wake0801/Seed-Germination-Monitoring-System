#!/usr/bin/env python3
"""Create a small demo time-lapse video from a GermPred test sequence."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPECIES_IMAGE_DIRS = {
    "pg": PROJECT_ROOT / "data/raw/GermPredDataset/PennisetumGlaucum/img",
    "sc": PROJECT_ROOT / "data/raw/GermPredDataset/SecaleCereale/img",
    "zm": PROJECT_ROOT / "data/raw/GermPredDataset/ZeaMays/img",
}


def frame_number(path: Path) -> int:
    match = re.search(r"_img(\d+)\.jpg$", path.name)
    return int(match.group(1)) if match else 0


def infer_species_code(sequence_id: str) -> str:
    match = re.match(r"^[a-z]+", sequence_id)
    if not match:
        raise ValueError(f"Cannot infer species code from sequence_id={sequence_id!r}")
    return match.group(0)


def load_sequence_frames(sequence_id: str) -> list[Path]:
    species_code = infer_species_code(sequence_id)
    image_dir = SPECIES_IMAGE_DIRS.get(species_code)
    if image_dir is None:
        raise ValueError(f"Unsupported species code {species_code!r}")
    frames = sorted(image_dir.glob(f"{sequence_id}_img*.jpg"), key=frame_number)
    if not frames:
        raise FileNotFoundError(f"No frames found for {sequence_id} in {image_dir}")
    return frames


def create_video(sequence_id: str, output_path: Path, fps: float, width: int) -> None:
    frames = load_sequence_frames(sequence_id)
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Cannot read frame: {frames[0]}")

    source_height, source_width = first.shape[:2]
    scale = width / source_width if width > 0 else 1.0
    output_size = (int(source_width * scale), int(source_height * scale))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        output_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer for {output_path}")

    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Cannot read frame: {frame_path}")
            if output_size != (source_width, source_height):
                frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-id", default="pg1_1", help="GermPred sequence ID to render.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/demo/videos/pg1_1_timelapse.mp4",
        help="Output MP4 path.",
    )
    parser.add_argument("--fps", type=float, default=8.0, help="Output video FPS.")
    parser.add_argument("--width", type=int, default=960, help="Resize video width; use 0 to keep source size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    create_video(args.sequence_id, output_path, args.fps, args.width)


if __name__ == "__main__":
    main()
