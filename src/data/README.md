# Pipeline Xử Lý Dữ Liệu

Thư mục này chứa các script xử lý dữ liệu cho baseline Deep Learning.

## File Chính

- `build_crops.py`: đọc XML, lấy bounding box, crop từng hạt giống từ ảnh gốc và tạo metadata.
- `split_by_sequence.py`: chia train/val/test theo `sequence_id` để tránh rò rỉ dữ liệu.

## Input

```text
data/raw/GermPredDataset/
```

Input gồm ảnh `.jpg` và annotation `.xml` theo chuẩn Pascal VOC.

## Output

```text
data/crops/
data/metadata/
```

Output gồm ảnh crop đã chia train/val/test và các file CSV mô tả dữ liệu.

## Lưu Ý

Pipeline xử lý dữ liệu chỉ phục vụ tạo input cho Deep Learning. Không thêm các thuật toán Machine Learning cổ điển hoặc đặc trưng thủ công vào phần này.
