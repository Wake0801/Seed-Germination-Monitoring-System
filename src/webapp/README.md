# Web Backend

Thư mục này chứa backend tối thiểu cho dashboard `web/`.

## Chạy Backend

```text
uvicorn src.webapp.api:app --reload
```

Sau đó mở:

```text
http://127.0.0.1:8000/
```

## API Chính

- `GET /api/health`: kiểm tra backend.
- `GET /api/models`: đọc `configs/model_registry.json`.
- `POST /api/predict/crop`: upload một ảnh crop và chạy model `.pth` bằng PyTorch.
- `POST /api/predict/detect`: upload ảnh raw và chạy Faster R-CNN scratch để trả về bounding box + trạng thái hạt.
- `/demo/*`: phục vụ ảnh demo trong `data/demo/`.

## `.pth` Hay ONNX?

File `.pth` chạy được nếu hệ thống web có Python backend dùng PyTorch. Đây là hướng hiện tại của project.

Không cần train lại model chỉ để xuất ONNX. ONNX chỉ cần khi:

- muốn chạy inference trực tiếp trong browser,
- muốn dùng ONNX Runtime,
- hoặc deploy sang môi trường không dùng PyTorch.

Nếu cần ONNX, dùng:

```text
python src/inference/export_onnx.py --model-id custom_cnn_baseline
```

## Model Chính Hiện Tại

Hệ thống web hiện ưu tiên Faster R-CNN scratch cho ảnh raw:

```text
outputs/checkpoints/faster_rcnn_scratch/best_faster_rcnn_resnet50_fpn_scratch.pth
```

Model này không dùng pretrained weights. File `.pth` lớn hơn giới hạn GitHub
thông thường, nên checkpoint được giữ local hoặc lưu ngoài repo; code và report
vẫn được commit bình thường.
