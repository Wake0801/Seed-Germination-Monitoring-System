#!/usr/bin/env python3
"""Utilities for reading model registry configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY_PATH = Path("configs/model_registry.json")


def load_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_models(registry: dict[str, Any]) -> list[dict[str, Any]]:
    return list(registry.get("models", []))


def get_model_config(
    model_id: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    task_type: str | None = None,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    selected_model = model_id or get_selected_model_id(registry, task_type)

    for model in list_models(registry):
        if model.get("id") == selected_model and (
            task_type is None or model.get("task_type") == task_type
        ):
            return model

    raise KeyError(f"Model id not found in registry: {selected_model}")


def get_selected_model_id(registry: dict[str, Any], task_type: str | None = None) -> str | None:
    if task_type == "crop_classification":
        return registry.get("selected_crop_model") or registry.get("selected_model")
    if task_type == "object_detection":
        return registry.get("selected_detection_model") or registry.get("selected_model")
    return registry.get("selected_model")


def get_decision_layer(registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    registry = load_registry(registry_path)
    return dict(registry.get("decision_layer", {}))
