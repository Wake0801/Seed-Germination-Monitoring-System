# Pipeline Xử Lý Dữ Liệu

Thư mục này chứa các script xử lý dữ liệu cho baseline Deep Learning.

## File Chính

- `build_crops.py`: đọc XML, lấy bounding box, crop từng hạt giống từ ảnh gốc và tạo metadata.
- `split_by_sequence.py`: chia train/val/test theo `sequence_id` để tránh rò rỉ dữ liệu.

## Cách Chạy

Tạo lại metadata và crop dataset cho baseline Custom CNN:

```text
python src/data/build_crops.py --overwrite
```

Nếu chỉ muốn kiểm tra metadata và split trước khi ghi ảnh crop:

```text
python src/data/build_crops.py --metadata-only
```

Nếu đã có `data/metadata/all_objects.csv` và chỉ muốn chia lại split:

```text
python src/data/split_by_sequence.py
```

Nếu đã có `data/metadata/all_objects_with_split.csv` và chỉ muốn tạo lại ảnh crop:

```text
python src/data/build_crops.py --from-metadata --overwrite
```

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

Cấu trúc crop dùng được trực tiếp với `torchvision.datasets.ImageFolder`:

```text
data/crops/
+-- train/
|   +-- germinated/
|   +-- non_germinated/
+-- val/
|   +-- germinated/
|   +-- non_germinated/
+-- test/
    +-- germinated/
    +-- non_germinated/
```

Các file metadata chính:

- `data/metadata/all_objects.csv`: toàn bộ object/bounding box trước khi gán split.
- `data/metadata/sequence_split.csv`: split ở mức `sequence_id`.
- `data/metadata/all_objects_with_split.csv`: toàn bộ object sau khi gán split.
- `data/metadata/train.csv`, `val.csv`, `test.csv`: metadata theo từng split.

## Trạng Thái Hiện Tại

Pipeline crop-classification đã chạy được trên toàn bộ `GermPredDataset`:

- Tổng object/crop: 235.933.
- Tổng ảnh gốc: 23.797.
- Tổng `sequence_id`: 247.
- Split không rò rỉ dữ liệu theo `sequence_id`.
- Ảnh crop đã được lưu theo cấu trúc `ImageFolder` trong `data/crops/`.

## Lưu Ý

Pipeline xử lý dữ liệu chỉ phục vụ tạo input cho Deep Learning. Không thêm các thuật toán Machine Learning cổ điển hoặc đặc trưng thủ công vào phần này.
