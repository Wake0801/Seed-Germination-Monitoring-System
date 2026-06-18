#!/usr/bin/env python3
"""FastAPI backend for the Seed Germination Monitoring web dashboard.

The frontend can run as static HTML, but this backend is the integration point
for real PyTorch `.pth` inference. ONNX is optional and only needed for
browser-side or ONNX Runtime deployment.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.inference.model_registry import DEFAULT_REGISTRY_PATH, load_registry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = PROJECT_ROOT / "web"

app = FastAPI(
    title="Seed Germination Monitoring API",
    description="Backend API for PyTorch checkpoint inference and dashboard integration.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def models() -> dict[str, Any]:
    registry = load_registry(DEFAULT_REGISTRY_PATH)
    return registry


@app.post("/api/predict/crop")
async def predict_crop(
    file: UploadFile = File(...),
    model_id: str | None = Form(default=None),
) -> dict[str, Any]:
    from src.inference.predict_one import predict_image

    suffix = Path(file.filename or "crop.jpg").suffix or ".jpg"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            shutil.copyfileobj(file.file, temp_file)

        result = predict_image(
            image_path=temp_path,
            model_id=model_id,
            registry_path=DEFAULT_REGISTRY_PATH,
            device_name="cpu",
        )
        return result
    except Exception as exc:  # pragma: no cover - keeps API errors readable
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            file.file.close()
        finally:
            if "temp_path" in locals():
                temp_path.unlink(missing_ok=True)


@app.post("/api/predict/detect")
async def predict_detect(
    file: UploadFile = File(...),
    model_id: str | None = Form(default=None),
    score_threshold: float | None = Form(default=None),
) -> dict[str, Any]:
    from src.inference.predict_detection import predict_detections

    suffix = Path(file.filename or "frame.jpg").suffix or ".jpg"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            shutil.copyfileobj(file.file, temp_file)

        result = predict_detections(
            image_path=temp_path,
            model_id=model_id,
            registry_path=DEFAULT_REGISTRY_PATH,
            score_threshold=score_threshold,
            device_name="cpu",
        )
        return result
    except Exception as exc:  # pragma: no cover - keeps API errors readable
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            file.file.close()
        finally:
            if "temp_path" in locals():
                temp_path.unlink(missing_ok=True)


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


app.mount("/assets", StaticFiles(directory=WEB_ROOT / "assets"), name="assets")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.webapp.api:app", host="127.0.0.1", port=8000, reload=True)
