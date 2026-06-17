# Tổng Quan Baseline Custom CNN

Tài liệu này tóm tắt lại hướng triển khai hiện tại sau khi đưa dự án bám sát README và phân công công việc nhóm. Baseline chính của đồ án là phân loại ảnh crop của từng hạt giống bằng Custom CNN, không dùng các mô hình Machine Learning cổ điển.

## Mục Tiêu

Mục tiêu giai đoạn baseline là xây dựng pipeline Deep Learning đơn giản, dễ giải thích và có thể chạy lại ổn định:

1. Đọc ảnh gốc và annotation Pascal VOC XML.
2. Lấy bounding box của từng hạt giống.
3. Chuẩn hóa nhãn về 2 lớp:
   - `*_im` -> `non_germinated`
   - `*_el` -> `germinated`
4. Crop từng hạt theo bounding box.
5. Chia train/validation/test theo `sequence_id` để tránh data leakage.
6. Train Custom CNN phân loại 2 lớp trên ảnh crop.

## Xử Lý Dữ Liệu

Script chính:

```text
src/data/build_crops.py
src/data/split_by_sequence.py
```

Kết quả đã tạo:

- Tổng ảnh gốc: 23.797.
- Tổng object/bounding box/crop: 235.933.
- Tổng sequence: 247.
- Số lỗi parse XML/bounding box/label: 0.
- Split được chia theo `sequence_id`, mỗi sequence chỉ nằm trong một tập duy nhất.

Phân bố sau split:

| Split | Germinated | Non germinated | Tổng crop |
| --- | ---: | ---: | ---: |
| Train | 89.436 | 99.772 | 189.208 |
| Validation | 10.254 | 12.478 | 22.732 |
| Test | 10.850 | 13.143 | 23.993 |
| Tổng | 110.540 | 125.393 | 235.933 |

Dataset crop được lưu theo cấu trúc tương thích với `torchvision.datasets.ImageFolder`:

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

- `data/metadata/all_objects.csv`
- `data/metadata/all_objects_with_split.csv`
- `data/metadata/sequence_split.csv`
- `data/metadata/train.csv`
- `data/metadata/val.csv`
- `data/metadata/test.csv`

## Cách Chạy Lại Data Pipeline

Tạo lại toàn bộ metadata và crop dataset:

```text
python src/data/build_crops.py --overwrite
```

Nếu metadata split đã có sẵn và chỉ muốn tạo lại ảnh crop:

```text
python src/data/build_crops.py --from-metadata --overwrite
```

Nếu chỉ muốn kiểm tra metadata/split, chưa ghi crop:

```text
python src/data/build_crops.py --metadata-only
```

## Custom CNN Baseline

File model:

```text
src/models/custom_cnn.py
```

Kiến trúc hiện tại:

- Input: ảnh RGB crop, resize về `224 x 224`.
- Feature extractor: 3 block `Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d`.
- Số kênh: `3 -> 32 -> 64 -> 128`.
- Pooling cuối: `AdaptiveAvgPool2d(1, 1)`.
- Classifier: `Dropout -> Linear(128, 64) -> ReLU -> Dropout -> Linear(64, 2)`.
- Output: logits 2 lớp, dùng với `CrossEntropyLoss`.
- Số tham số trainable mặc định: 101.858.

Model đã được kiểm tra forward pass với input `[4, 3, 224, 224]` và trả output `[4, 2]`.

## Kết Quả Train Baseline

Baseline đã được train bằng notebook Colab:

```text
notebooks/custom_cnn_colab.ipynb
```

Kết quả và checkpoint được sắp xếp theo đúng cấu trúc `outputs/`:

```text
outputs/checkpoints/baseline_cnn/best_custom_cnn.pth
outputs/logs/baseline_cnn/training_history.csv
outputs/reports/baseline_cnn/test_summary.json
outputs/reports/baseline_cnn/test_metrics.csv
outputs/figures/baseline_cnn/learning_curves.png
outputs/figures/baseline_cnn/confusion_matrix.png
```

Các file kết quả chính:

- `outputs/checkpoints/baseline_cnn/best_custom_cnn.pth`: checkpoint tốt nhất theo validation macro F1.
- `outputs/logs/baseline_cnn/training_history.csv`: log train/validation theo epoch.
- `outputs/reports/baseline_cnn/test_summary.json`: metric tổng hợp trên test set.
- `outputs/reports/baseline_cnn/test_metrics.csv`: metric theo từng lớp.
- `outputs/figures/baseline_cnn/learning_curves.png`: biểu đồ loss và macro F1.
- `outputs/figures/baseline_cnn/confusion_matrix.png`: confusion matrix trên test set.

Kết quả test:

| Metric | Giá trị |
| --- | ---: |
| Accuracy | 0.8336 |
| Macro Precision | 0.8327 |
| Macro Recall | 0.8308 |
| Macro F1-score | 0.8316 |
| Weighted F1-score | 0.8334 |

Metric theo từng lớp:

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| `germinated` | 0.8253 | 0.8018 | 0.8134 | 10.850 |
| `non_germinated` | 0.8402 | 0.8598 | 0.8499 | 13.143 |

Nhận xét nhanh:

- Mô hình đạt macro F1 khoảng `0.8316`, đủ dùng làm mốc baseline đầu tiên.
- Lớp `non_germinated` có F1 cao hơn lớp `germinated`.
- Recall của lớp `germinated` thấp hơn, cho thấy mô hình vẫn bỏ sót một phần hạt đã nảy mầm.
- Bước tiếp theo nên phân tích các ảnh dự đoán sai và so sánh với Transfer Learning.

## Vì Sao Không Lấy YOLO Là Baseline Chính

YOLO/object detection là hướng Deep Learning hợp lý cho ứng dụng thực tế khi cần xử lý ảnh nguyên đĩa petri. Tuy nhiên, theo README và phân công ban đầu, baseline cần ưu tiên crop-classification bằng Custom CNN vì:

- Dễ giải thích kiến trúc Deep Learning tự xây.
- Phù hợp với nhiệm vụ của Người 1 và Người 2.
- Dễ so sánh với Transfer Learning ở giai đoạn cải tiến.
- Giữ pipeline đơn giản trước khi mở rộng sang object detection.

Các script YOLO/COCO trong `src/data_preprocessing/` có thể giữ lại như hướng mở rộng sau baseline, nhưng không phải pipeline chính của giai đoạn hiện tại.

## Bước Tiếp Theo

Các bước tiếp theo sau baseline:

- Đưa kết quả baseline vào báo cáo.
- Demo dự đoán trên một số ảnh trong `data/crops/test/`.
- Phân tích ảnh bị dự đoán sai, đặc biệt các mẫu `germinated` bị nhầm sang `non_germinated`.
- Train thử Transfer Learning như ResNet18, MobileNetV2 hoặc EfficientNet-B0.
- So sánh Transfer Learning với Custom CNN bằng cùng test set.
