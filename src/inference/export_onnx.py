#!/usr/bin/env python3
"""Export a registered PyTorch checkpoint to ONNX.

This is optional. Keep `.pth` for PyTorch backend inference; export ONNX only
when deploying to a runtime that needs it, such as ONNX Runtime or browser-side
inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from src.inference.model_registry import DEFAULT_REGISTRY_PATH, get_model_config
    from src.inference.predict_one import load_registered_model, resolve_path
except ModuleNotFoundError:  # Allows running this file directly from src/inference.
    from model_registry import DEFAULT_REGISTRY_PATH, get_model_config
    from predict_one import load_registered_model, resolve_path


def export_registered_model(
    model_id: str | None,
    output_path: str | Path,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> Path:
    device = torch.device("cpu")
    model, _, model_config = load_registered_model(model_id, registry_path, device)
    image_size = int(model_config.get("input_size", 224))
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    destination = resolve_path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        destination,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a registered model to ONNX.")
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--output",
        default="outputs/checkpoints/baseline_cnn/best_custom_cnn.onnx",
    )
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    args = parser.parse_args()

    model_config = get_model_config(args.model_id, args.registry)
    output = export_registered_model(
        model_id=model_config["id"],
        output_path=args.output,
        registry_path=args.registry,
    )
    print(f"Exported ONNX model: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
