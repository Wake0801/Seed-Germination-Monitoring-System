# Dữ Liệu Đã Xử Lý

Thư mục này mô tả dữ liệu sau bước tiền xử lý. Trong baseline hiện tại, dữ liệu huấn luyện chính là ảnh crop của từng hạt giống được tạo từ bounding box trong XML.

## Kết Quả Mong Đợi

Sau khi chạy pipeline xử lý dữ liệu, cấu trúc chính sẽ gồm:

```text
data/crops/
├── train/
│   ├── non_germinated/
│   └── germinated/
├── val/
│   ├── non_germinated/
│   └── germinated/
└── test/
    ├── non_germinated/
    └── germinated/
```

Metadata dự kiến nằm trong:

```text
data/metadata/
├── all_objects.csv
├── sequence_split.csv
├── train.csv
├── val.csv
└── test.csv
```

## Nguyên Tắc Xử Lý

- Mỗi bounding box trong XML tạo ra một ảnh crop hạt giống.
- Nhãn được chuẩn hóa về 2 lớp: `non_germinated` và `germinated`.
- Train/val/test phải được chia theo `sequence_id`.
- Không để các frame thuộc cùng một sequence xuất hiện ở nhiều tập khác nhau.

## Vai Trò

Dữ liệu đã xử lý là đầu vào trực tiếp cho mô hình Deep Learning. Người xây model và người training chỉ nên dùng dữ liệu ở `data/crops/` và metadata ở `data/metadata/`, không cần đọc trực tiếp XML gốc.
