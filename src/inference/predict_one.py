#!/usr/bin/env python3
"""Predict one cropped seed image with a registered PyTorch checkpoint.

The model still learns the original two dataset classes:
  - germinated
  - non_germinated

The optional third display state, ``transition``, is produced by the decision
layer from softmax confidence. It is not a third trained class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch import nn

try:
    from src.inference.model_registry import (
        DEFAULT_REGISTRY_PATH,
        get_decision_layer,
        get_model_config,
    )
except ModuleNotFoundError:  # Allows running this file directly from src/inference.
    from model_registry import DEFAULT_REGISTRY_PATH, get_decision_layer, get_model_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def import_project_models() -> None:
    import sys

    project_root_text = str(PROJECT_ROOT)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)


def build_model(architecture: str, num_classes: int) -> nn.Module:
    import_project_models()

    if architecture == "custom_cnn":
        from src.models.custom_cnn import build_custom_cnn

        return build_custom_cnn(num_classes=num_classes)

    raise ValueError(f"Unsupported model architecture: {architecture}")


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_checkpoint(path: str | Path, map_location: str) -> dict[str, Any]:
    checkpoint_path = resolve_path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict):
        return checkpoint

    raise TypeError("Expected checkpoint dict with model_state_dict.")


def load_registered_model(
    model_id: str | None,
    registry_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, list[str], dict[str, Any]]:
    model_config = get_model_config(
        model_id=model_id,
        registry_path=registry_path,
        task_type="crop_classification",
    )
    if model_config.get("status") != "ready":
        raise RuntimeError(
            f"Model '{model_config.get('id')}' is not ready. "
            "Update configs/model_registry.json after adding its checkpoint."
        )

    checkpoint = load_checkpoint(model_config["checkpoint_path"], map_location=str(device))
    class_names = checkpoint.get("class_names") or model_config.get("class_names")
    if not class_names:
        raise ValueError("Class names must be present in checkpoint or model registry.")

    model = build_model(
        architecture=str(model_config["architecture"]),
        num_classes=int(model_config.get("num_classes", len(class_names))),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, list(class_names), model_config


def image_to_tensor(image_path: str | Path, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)

    values = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = values.reshape(image_size, image_size, 3).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.unsqueeze(0)


def apply_decision_layer(
    probabilities: dict[str, float],
    decision_layer: dict[str, Any],
) -> str:
    if not decision_layer.get("enabled", False):
        return max(probabilities, key=probabilities.get)

    positive_class = decision_layer.get("positive_class", "germinated")
    positive_probability = probabilities.get(positive_class, 0.0)
    germinated_threshold = float(decision_layer.get("germinated_threshold", 0.7))
    non_germinated_threshold = float(decision_layer.get("non_germinated_threshold", 0.3))

    if positive_probability >= germinated_threshold:
        return "germinated"
    if positive_probability <= non_germinated_threshold:
        return "non_germinated"
    return "transition"


def predict_image(
    image_path: str | Path,
    model_id: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    device_name: str | None = None,
) -> dict[str, Any]:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, class_names, model_config = load_registered_model(model_id, registry_path, device)
    image_size = int(model_config.get("input_size", 224))
    image_tensor = image_to_tensor(image_path, image_size).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()

    probabilities = {
        class_name: float(probability)
        for class_name, probability in zip(class_names, probs)
    }
    predicted_class = max(probabilities, key=probabilities.get)
    display_state = apply_decision_layer(
        probabilities,
        get_decision_layer(registry_path),
    )

    return {
        "model_id": model_config.get("id"),
        "architecture": model_config.get("architecture"),
        "image_path": str(image_path),
        "predicted_class": predicted_class,
        "display_state": display_state,
        "confidence": probabilities[predicted_class],
        "probabilities": probabilities,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict one cropped seed image.")
    parser.add_argument("image_path")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--device", default=None, help="cpu, cuda, or omitted for auto")
    args = parser.parse_args()

    result = predict_image(
        image_path=args.image_path,
        model_id=args.model_id,
        registry_path=args.registry,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
