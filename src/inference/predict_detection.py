#!/usr/bin/env python3
"""Run object detection on a raw seed image with Faster R-CNN.

This module is separate from ``predict_one.py`` because the baseline predicts a
single cropped seed, while Faster R-CNN predicts multiple boxes on a raw image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch

try:
    from src.inference.model_registry import DEFAULT_REGISTRY_PATH, get_model_config
except ModuleNotFoundError:  # Allows running this file directly from src/inference.
    from model_registry import DEFAULT_REGISTRY_PATH, get_model_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def import_project_models() -> None:
    import sys

    project_root_text = str(PROJECT_ROOT)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def image_to_tensor(image_path: str | Path) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    values = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = values.reshape(height, width, 3).permute(2, 0, 1).float() / 255.0
    return tensor, (width, height)


def normalize_class_names(raw_class_names: Any) -> dict[int, str]:
    if isinstance(raw_class_names, dict):
        return {int(key): str(value) for key, value in raw_class_names.items()}
    if isinstance(raw_class_names, list):
        return {index: str(value) for index, value in enumerate(raw_class_names)}
    raise ValueError("Class names must be a dict or list.")


def load_checkpoint(path: str | Path, map_location: str) -> dict[str, Any]:
    checkpoint_path = resolve_path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected checkpoint dict.")
    return checkpoint


def load_registered_detection_model(
    model_id: str | None,
    registry_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[int, str], dict[str, Any]]:
    import_project_models()
    from src.models.faster_rcnn import build_detection_model

    model_config = get_model_config(
        model_id=model_id,
        registry_path=registry_path,
        task_type="object_detection",
    )
    if model_config.get("status") != "ready":
        raise RuntimeError(
            f"Model '{model_config.get('id')}' is not ready. "
            "Update configs/model_registry.json after adding its checkpoint."
        )

    checkpoint = load_checkpoint(model_config["checkpoint_path"], map_location=str(device))
    raw_class_names = checkpoint.get("class_names") or model_config.get("class_names")
    class_names = normalize_class_names(raw_class_names)

    checkpoint_config = checkpoint.get("config", {})
    min_size = int(model_config.get("min_size") or checkpoint_config.get("min_size") or 640)
    max_size = int(model_config.get("max_size") or checkpoint_config.get("max_size") or min_size)

    model = build_detection_model(
        architecture=str(model_config["architecture"]),
        num_classes=int(model_config.get("num_classes", len(class_names))),
        min_size=min_size,
        max_size=max_size,
    )
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint must contain 'state_dict' or 'model_state_dict'.")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, class_names, model_config


def clamp_box(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width), float(x1)))
    y1 = max(0.0, min(float(height), float(y1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y2 = max(0.0, min(float(height), float(y2)))
    return x1, y1, x2, y2


def format_detections(
    output: dict[str, torch.Tensor],
    class_names: dict[int, str],
    image_size: tuple[int, int],
    score_threshold: float,
) -> list[dict[str, Any]]:
    width, height = image_size
    boxes = output["boxes"].detach().cpu().tolist()
    labels = output["labels"].detach().cpu().tolist()
    scores = output["scores"].detach().cpu().tolist()

    detections: list[dict[str, Any]] = []
    for index, (box, label, score) in enumerate(zip(boxes, labels, scores), start=1):
        score_value = float(score)
        if score_value < score_threshold:
            continue

        x1, y1, x2, y2 = clamp_box(box, width, height)
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        class_name = class_names.get(int(label), str(label))

        detections.append(
            {
                "id": len(detections) + 1,
                "raw_index": index,
                "class_id": int(label),
                "class_name": class_name,
                "display_state": class_name,
                "score": score_value,
                "confidence": round(score_value * 100, 2),
                "bbox": {
                    "x": round(x1, 2),
                    "y": round(y1, 2),
                    "width": round(box_width, 2),
                    "height": round(box_height, 2),
                },
                "bbox_percent": {
                    "x": round((x1 / width) * 100, 4) if width else 0,
                    "y": round((y1 / height) * 100, 4) if height else 0,
                    "width": round((box_width / width) * 100, 4) if width else 0,
                    "height": round((box_height / height) * 100, 4) if height else 0,
                },
            }
        )

    return detections


def predict_detections(
    image_path: str | Path,
    model_id: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    score_threshold: float | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, class_names, model_config = load_registered_detection_model(
        model_id=model_id,
        registry_path=registry_path,
        device=device,
    )
    threshold = float(score_threshold or model_config.get("score_threshold", 0.5))
    image_tensor, image_size = image_to_tensor(image_path)

    with torch.no_grad():
        output = model([image_tensor.to(device)])[0]

    detections = format_detections(
        output=output,
        class_names=class_names,
        image_size=image_size,
        score_threshold=threshold,
    )

    return {
        "model_id": model_config.get("id"),
        "architecture": model_config.get("architecture"),
        "image_path": str(image_path),
        "image_size": {
            "width": image_size[0],
            "height": image_size[1],
        },
        "score_threshold": threshold,
        "detections": detections,
        "summary": summarize_detections(detections),
    }


def summarize_detections(detections: list[dict[str, Any]]) -> dict[str, int]:
    germinated = sum(item["display_state"] == "germinated" for item in detections)
    non_germinated = sum(item["display_state"] == "non_germinated" for item in detections)
    transition = sum(item["display_state"] == "transition" for item in detections)
    return {
        "total": len(detections),
        "germinated": germinated,
        "non_germinated": non_germinated,
        "transition": transition,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect seeds in one raw image.")
    parser.add_argument("image_path")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--device", default=None, help="cpu, cuda, or omitted for auto")
    args = parser.parse_args()

    result = predict_detections(
        image_path=args.image_path,
        model_id=args.model_id,
        registry_path=args.registry,
        score_threshold=args.score_threshold,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
