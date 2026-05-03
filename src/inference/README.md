# Dự Đoán

Thư mục này chứa script inference sau khi đã có checkpoint mô hình.

## File Chính

- `predict_one.py`: load checkpoint Deep Learning và dự đoán trạng thái của một ảnh crop hạt giống.

## Input Dự Kiến

```text
image_path
checkpoint_path
model_name
```

## Output Dự Kiến

Kết quả nên được trả về ở dạng dễ tích hợp, ví dụ JSON:

```json
{
  "label": "germinated",
  "confidence": 0.95
}
```

## Vai Trò Sau Này

Phần này sẽ được dùng để tích hợp với Web Backend. Spring Boot có thể gọi script Python bằng `ProcessBuilder` và đọc kết quả dự đoán.
