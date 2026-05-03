# Hệ Thống Giám Sát Nảy Mầm Hạt Giống

Đây là cấu trúc tạm thời cho phần Deep Learning của đồ án giám sát nảy mầm hạt giống. Mục tiêu trước mắt là xây dựng một baseline đơn giản, dễ chạy, dễ giải thích và có thể mở rộng sang các mô hình mạnh hơn sau này.

## Phạm Vi Hiện Tại

- Bài toán baseline: phân loại ảnh crop của từng hạt thành 2 lớp `non_germinated` và `germinated`.
- Phương pháp chính: Deep Learning 100%, bắt đầu bằng Custom CNN.
- Dữ liệu đầu vào: ảnh `.jpg` và annotation `.xml` trong `data/raw/GermPredDataset/`.
- Cách chia dữ liệu: chia theo `sequence_id`, không chia ngẫu nhiên từng ảnh để tránh rò rỉ dữ liệu.
- Không dùng các thuật toán Machine Learning cổ điển như SVM, Random Forest, KNN, Logistic Regression, Decision Tree hoặc các đặc trưng thủ công như HOG/LBP làm phương pháp chính.

## Cấu Trúc Chính

- `configs/`: lưu cấu hình cho baseline CNN và các thử nghiệm cải tiến.
- `data/`: lưu dataset raw, crop dataset, metadata và dữ liệu đã xử lý.
- `docs/`: lưu ghi chú dữ liệu và phân công công việc nhóm.
- `src/data/`: chứa script xử lý dữ liệu, parse XML, crop ảnh và chia tập.
- `src/models/`: chứa định nghĩa các mô hình Deep Learning.
- `src/training/`: chứa script huấn luyện và đánh giá mô hình.
- `src/inference/`: chứa script dự đoán, phục vụ tích hợp hệ thống sau này.
- `outputs/`: lưu checkpoint, log, biểu đồ và báo cáo kết quả.
- `notebooks/`: dùng cho phân tích nhanh dataset hoặc kiểm tra lỗi mô hình, không dùng thay thế pipeline chính.

## Dataset

Dataset hiện được đặt tại:

```text
data/raw/GermPredDataset/
```

Dataset gồm 3 loài hạt:

- `PennisetumGlaucum`
- `SecaleCereale`
- `ZeaMays`

Mỗi loài có 2 thư mục:

- `img/`: ảnh `.jpg`
- `true_ann/`: nhãn `.xml` theo chuẩn Pascal VOC

Raw dataset được ignore khỏi Git vì có kích thước lớn. Repository chỉ lưu cấu trúc, tài liệu, config và mã nguồn.

## Hướng Làm Baseline

1. Đọc XML và ảnh gốc.
2. Map nhãn `*_im` thành `non_germinated`, `*_el` thành `germinated`.
3. Crop từng hạt theo bounding box.
4. Chia train/val/test theo `sequence_id`.
5. Train Custom CNN đơn giản.
6. Đánh giá bằng Accuracy, Precision, Recall, F1-score và Confusion Matrix.
