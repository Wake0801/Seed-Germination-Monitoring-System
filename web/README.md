# Seed Germination Monitoring Web

Thư mục này chứa giao diện dashboard cho hệ thống giám sát nảy mầm hạt giống.

## Cấu Trúc

```text
web/
+-- index.html
+-- assets/
    +-- css/dashboard.css
    +-- js/data.js
    +-- js/api.js
    +-- js/ui.js
    +-- js/app.js
```

`index.html` chỉ giữ layout chính. CSS, dữ liệu mô phỏng, API adapter và logic UI đã được tách riêng để dễ bảo trì.

## Chạy Giao Diện Tĩnh

Mở trực tiếp file:

```text
web/index.html
```

Hoặc chạy server tĩnh bằng Python:

```text
python -m http.server 8000
```

Sau đó mở:

```text
http://localhost:8000/web/
```

## Cấu Trúc Giao Diện

Dashboard có 2 phần chính:

- `Monitoring System`: upload ảnh raw và xem bounding box + trạng thái hạt từ Faster R-CNN scratch.
- `Model Comparison`: so sánh baseline Custom CNN với Faster R-CNN scratch theo metric phù hợp.

Khi backend chạy, giao diện gọi inference thật trong `src/inference/`. Nếu backend chưa chạy, giao diện vẫn có dữ liệu mô phỏng để kiểm tra UI.

Ảnh demo có sẵn tại:

```text
data/demo/raw_images/
```

Các ảnh này thuộc test split và được web phục vụ qua route `/demo`.

## Chạy Với Backend PyTorch

Nếu muốn dùng checkpoint `.pth` thật, chạy backend:

```text
uvicorn src.webapp.api:app --reload
```

Sau đó mở:

```text
http://127.0.0.1:8000/
```

Backend cung cấp:

- `GET /api/models`
- `POST /api/predict/crop`
- `POST /api/predict/detect`

Frontend sẽ cố gọi API model registry và endpoint Faster R-CNN detection. Nếu backend chưa chạy, giao diện vẫn hoạt động bằng dữ liệu mô phỏng.

## Kết Nối Model

Model được khai báo trong:

```text
configs/model_registry.json
```

Model detection chính hiện tại:

```text
outputs/checkpoints/faster_rcnn_scratch/best_faster_rcnn_resnet50_fpn_scratch.pth
```

Baseline crop classification vẫn được giữ tại:

```text
outputs/checkpoints/baseline_cnn/best_custom_cnn.pth
```

## Lưu Ý

Model trạng thái học 2 lớp chính của dataset:

```text
germinated
non_germinated
```

Trạng thái `transition` trên giao diện là trạng thái vận hành được suy ra từ confidence threshold, không phải lớp thứ ba được train trực tiếp.

File `.pth` chạy được với backend Python/PyTorch. Không cần export ONNX trừ khi muốn chạy inference trực tiếp trong browser hoặc ONNX Runtime.
