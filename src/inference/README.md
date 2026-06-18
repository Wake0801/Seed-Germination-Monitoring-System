# Dự Đoán

Thư mục này chứa script inference sau khi đã có checkpoint mô hình.

## File Chính

- `predict_one.py`: load checkpoint Deep Learning và dự đoán trạng thái của một ảnh crop hạt giống.
- `predict_detection.py`: load checkpoint Faster R-CNN scratch và phát hiện nhiều hạt trên ảnh raw.
- `model_registry.py`: đọc danh sách model từ `configs/model_registry.json`.
- `export_onnx.py`: xuất checkpoint PyTorch sang ONNX khi cần deploy bằng runtime khác.

## Input Dự Kiến

```text
image_path
model_id
configs/model_registry.json
```

## Output Dự Kiến

Kết quả nên được trả về ở dạng dễ tích hợp, ví dụ JSON:

```json
{
  "model_id": "custom_cnn_baseline",
  "architecture": "custom_cnn",
  "image_path": "data/crops/test/germinated/example.jpg",
  "predicted_class": "germinated",
  "display_state": "germinated",
  "confidence": 0.95,
  "probabilities": {
    "germinated": 0.95,
    "non_germinated": 0.05
  }
}
```

## Định Dạng Model

Checkpoint `.pth` hiện tại là đúng cho hệ thống backend dùng PyTorch. File này lưu:

- `model_state_dict`
- `class_names`
- `class_to_idx`
- `config`
- `val_summary`

ONNX không bắt buộc ở giai đoạn hiện tại. Chỉ nên export ONNX nếu muốn chạy model bằng ONNX Runtime, browser inference hoặc backend không dùng PyTorch.

Lệnh export ONNX khi cần:

```text
python src/inference/export_onnx.py --model-id custom_cnn_baseline
```

## Tích Hợp Web

Phần này đã được dùng bởi backend FastAPI trong:

```text
src/webapp/api.py
```

Dashboard web có thể gọi `POST /api/predict/crop` để chạy checkpoint `.pth` bằng PyTorch.
Với hệ thống chính hiện tại, dashboard gọi `POST /api/predict/detect` để nhận danh sách bounding box và trạng thái từng hạt từ Faster R-CNN scratch.

Nếu dùng hệ thống khác như Spring Boot, vẫn có thể gọi script Python bằng `ProcessBuilder` và đọc kết quả JSON.
