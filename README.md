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

## Trạng Thái Hiện Tại

Phần xử lý dữ liệu và baseline crop-classification đã chạy được end-to-end:

- `src/data/build_crops.py`: tạo metadata object-level, split theo `sequence_id` và xuất ảnh crop.
- `src/data/split_by_sequence.py`: chia lại metadata theo `sequence_id` khi cần.
- `data/crops/`: chứa crop dataset theo cấu trúc `ImageFolder`.
- `data/metadata/`: chứa `all_objects.csv`, `all_objects_with_split.csv`, `sequence_split.csv`, `train.csv`, `val.csv`, `test.csv`.
- `src/models/custom_cnn.py`: đã có Custom CNN baseline 3 block convolution, output logits 2 lớp.
- `notebooks/custom_cnn_colab.ipynb`: notebook Colab dùng để train/validate/test baseline.
- `outputs/checkpoints/baseline_cnn/`: chứa checkpoint tốt nhất của baseline.
- `outputs/logs/baseline_cnn/`: chứa log train/validation theo epoch.
- `outputs/reports/baseline_cnn/`: chứa metric test và README tóm tắt kết quả baseline.
- `outputs/figures/baseline_cnn/`: chứa learning curves và confusion matrix.

Kết quả test baseline hiện tại:

| Metric | Giá trị |
| --- | ---: |
| Accuracy | 0.8336 |
| Macro Precision | 0.8327 |
| Macro Recall | 0.8308 |
| Macro F1-score | 0.8316 |
| Weighted F1-score | 0.8334 |

Cách tạo lại crop dataset:

```text
python src/data/build_crops.py --overwrite
```

Nếu metadata đã có sẵn và chỉ muốn tạo lại ảnh crop:

```text
python src/data/build_crops.py --from-metadata --overwrite
```

Lưu ý: baseline được train bằng Colab notebook vì máy local không phù hợp để train toàn bộ crop dataset. Các file `src/training/train_baseline.py` và `src/training/evaluate.py` chưa dùng làm pipeline chính.

Bước tiếp theo của đồ án là phân tích lỗi baseline, demo dự đoán trên test set và cải tiến bằng Transfer Learning.
