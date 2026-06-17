# Hướng Phát Triển Đồ Án

Tài liệu này mô tả hướng phát triển tiếp theo của đồ án giám sát nảy mầm hạt giống bằng Deep Learning. Baseline phân loại ảnh crop bằng Custom CNN đã được train bằng Colab notebook, vì vậy trọng tâm tiếp theo là phân tích kết quả, chuẩn bị demo và mở rộng sang mô hình mạnh hơn.

## 1. Trạng Thái Hiện Tại

Đồ án hiện đang ở giai đoạn đã hoàn thành tiền xử lý dữ liệu và đã có kết quả baseline đầu tiên.

Các phần đã có:

- Raw dataset GermPredDataset đã được đặt trong `data/raw/GermPredDataset/`.
- Annotation Pascal VOC XML đã được parse để lấy bounding box từng hạt.
- Nhãn gốc đã được chuẩn hóa về 2 lớp:
  - `*_im` -> `non_germinated`
  - `*_el` -> `germinated`
- Dataset crop đã được tạo theo cấu trúc `ImageFolder`:

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

- Dữ liệu đã được chia theo `sequence_id` để tránh rò rỉ dữ liệu giữa các frame cùng một đĩa Petri.
- Model baseline `CustomCNN` đã được định nghĩa trong `src/models/custom_cnn.py`.
- Notebook Colab `notebooks/custom_cnn_colab.ipynb` đã được bổ sung các cell train, validate, test, lưu checkpoint và xuất confusion matrix.
- Baseline đã được train bằng Colab notebook.
- Kết quả train baseline đã được sắp xếp trong `outputs/`:
  - `outputs/checkpoints/baseline_cnn/`
  - `outputs/logs/baseline_cnn/`
  - `outputs/reports/baseline_cnn/`
  - `outputs/figures/baseline_cnn/`

Kết quả test baseline hiện tại:

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

Các phần chưa hoàn thành:

- Chưa phân tích chi tiết các ảnh dự đoán sai.
- Chưa có demo ảnh nguyên đĩa Petri với bounding box XML.
- Chưa có mô hình Transfer Learning để so sánh với Custom CNN.
- Script local `src/training/train_baseline.py` và `src/training/evaluate.py` vẫn chưa được triển khai và không phải pipeline chính do giới hạn tài nguyên local.

## 2. Mục Tiêu Giai Đoạn Gần Nhất

Mục tiêu trước mắt là đóng gói kết quả baseline vào báo cáo và chuẩn bị demo.

Việc cần làm:

1. Đưa bảng metric baseline vào báo cáo.
2. Đưa `learning_curves.png` và `confusion_matrix.png` vào phần kết quả.
3. Viết nhận xét về kết quả: mô hình nhận diện `non_germinated` tốt hơn nhẹ so với `germinated`.
4. Demo dự đoán vài ảnh trong `data/crops/test/`.
5. Nếu cần demo trực quan hơn, tạo demo ảnh nguyên đĩa Petri bằng bounding box XML.
6. Sau khi báo cáo baseline ổn định, chuyển sang Transfer Learning.

## 3. Giai Đoạn 1: Baseline Custom CNN

Đây là giai đoạn quan trọng nhất hiện tại. Mục tiêu không phải đạt kết quả cao nhất ngay, mà là có một pipeline Deep Learning hoàn chỉnh, dễ giải thích và có thể chạy lại.

Input:

```text
ảnh crop của một hạt giống
```

Output:

```text
germinated hoặc non_germinated
```

Metric đánh giá:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Trạng thái hoàn thành:

- Model đã train được 1 lần hoàn chỉnh bằng Colab.
- Đã có checkpoint tốt nhất theo validation macro F1.
- Đã có kết quả test set.
- Đã có biểu đồ loss/F1 theo epoch.
- Đã có confusion matrix.
- Cần bổ sung phần nhận xét lỗi dự đoán trong báo cáo.

Nội dung báo cáo cần viết sau giai đoạn này:

- Mô tả kiến trúc Custom CNN.
- Mô tả cách chia dữ liệu theo `sequence_id`.
- Trình bày cấu hình train: image size, batch size, optimizer, learning rate, số epoch.
- Trình bày kết quả test.
- Nhận xét các trường hợp mô hình dễ nhầm lẫn.

## 4. Giai Đoạn 2: Cải Tiến Bằng Transfer Learning

Sau khi có baseline Custom CNN, hướng cải tiến hợp lý là dùng transfer learning trên cùng bài toán crop-classification.

Các model đề xuất:

- ResNet18
- MobileNetV2
- EfficientNet-B0

Lý do chọn transfer learning:

- Tận dụng feature extractor đã học từ ImageNet.
- Thường hội tụ nhanh hơn Custom CNN.
- Có cơ sở so sánh trực tiếp với baseline tự xây.
- Phù hợp với phạm vi đồ án hơn so với nhảy ngay sang object detection phức tạp.

Metric vẫn giống giai đoạn baseline:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Tiêu chí hoàn thành:

- Train ít nhất một backbone transfer learning.
- So sánh kết quả với Custom CNN.
- Nhận xét trade-off giữa độ chính xác, số tham số và thời gian train.

Bảng so sánh nên có:

| Model | Accuracy | Precision | Recall | F1-score | Ghi chú |
| --- | ---: | ---: | ---: | ---: | --- |
| Custom CNN | TBD | TBD | TBD | TBD | Baseline |
| ResNet18 | TBD | TBD | TBD | TBD | Transfer learning |
| MobileNetV2 | TBD | TBD | TBD | TBD | Nhẹ, dễ deploy |
| EfficientNet-B0 | TBD | TBD | TBD | TBD | Hiệu quả tham số tốt |

## 5. Giai Đoạn 3: Demo Hệ Thống

Do hiện tại chưa có ảnh tự chụp từ hệ thống camera, demo có thể dùng hold-out test set của GermPredDataset. Đây là cách hợp lệ vì test set không được dùng trong quá trình huấn luyện.

### Demo đơn giản

Input:

```text
ảnh crop của một hạt từ data/crops/test/
```

Output:

```text
nhãn thật, nhãn dự đoán, confidence
```

Ví dụ:

```text
Ground truth: germinated
Prediction: germinated
Confidence: 96.4%
```

### Demo nâng cao

Input:

```text
ảnh nguyên đĩa Petri từ data/raw/GermPredDataset/
```

Cách làm:

1. Đọc ảnh gốc.
2. Đọc bounding box từ XML tương ứng.
3. Crop từng hạt theo bounding box.
4. Dự đoán trạng thái từng crop bằng model đã train.
5. Vẽ bounding box lên ảnh gốc:
   - Một màu cho `germinated`
   - Một màu cho `non_germinated`

Lưu ý: Demo nâng cao vẫn dùng bounding box từ annotation có sẵn, chưa phải hệ thống object detection tự động hoàn toàn. Cần trình bày rõ giới hạn này trong báo cáo.

## 6. Giai Đoạn 4: Object Detection Cho Ảnh Nguyên Đĩa Petri

Baseline hiện tại chỉ phân loại ảnh crop. Nếu muốn hệ thống nhận ảnh nguyên đĩa Petri và tự tìm từng hạt, cần thêm object detection.

Input:

```text
ảnh nguyên đĩa Petri chứa nhiều hạt
```

Output:

```text
danh sách bounding box + trạng thái germinated/non_germinated
```

Hướng đề xuất:

- Faster R-CNN
- Faster R-CNN + FPN
- Mask R-CNN nếu cần phân đoạn chính xác hình dạng hạt/rễ

Không nên chuyển sang object detection trước khi có kết quả baseline, vì khi đó đồ án sẽ thiếu mốc so sánh ban đầu.

Metric đánh giá:

- mAP
- IoU
- Precision/Recall theo object
- Số lượng hạt dự đoán đúng/sai trên từng ảnh

Tiêu chí hoàn thành:

- Model phát hiện được hạt trên ảnh nguyên đĩa Petri.
- Model phân biệt được trạng thái nảy mầm/chưa nảy mầm.
- Có hình demo bounding box trên ảnh test.

## 7. Giai Đoạn 5: Theo Dõi Theo Thời Gian

Khi đã có mô hình nhận diện ổn định, có thể mở rộng sang bài toán giám sát quá trình nảy mầm theo thời gian.

Mục tiêu:

- Theo dõi trạng thái của hạt qua nhiều frame.
- Xác định thời điểm hạt chuyển từ `non_germinated` sang `germinated`.
- Tính các chỉ số sinh lý học nảy mầm.

Chỉ số có thể tính:

- Germination Percentage
- Mean Germination Time
- Germination Index
- Đường cong tích lũy nảy mầm theo thời gian

Vấn đề cần xử lý:

- Cùng một hạt xuất hiện ở nhiều frame.
- Dự đoán có thể bị dao động giữa hai nhãn.
- Cần smoothing hoặc quy tắc chuyển trạng thái một chiều:

```text
non_germinated -> germinated
```

Không nên cho phép chuyển ngược tùy tiện:

```text
germinated -> non_germinated
```

## 8. Thứ Tự Ưu Tiên Khuyến Nghị

Thứ tự nên làm tiếp để đồ án chắc chắn có kết quả tốt:

1. Viết phần kết quả baseline vào báo cáo.
2. Làm demo crop-level trên test set.
3. Phân tích confusion matrix và ảnh dự đoán sai.
4. Nếu còn thời gian, train transfer learning.
5. Nếu còn nhiều thời gian, làm demo ảnh nguyên đĩa Petri bằng bounding box XML.
6. Object detection và temporal tracking để ở phần mở rộng.

Không nên nhảy ngay sang Faster R-CNN, Mask R-CNN hoặc hệ thống video khi baseline chưa có kết quả thực nghiệm.

## 9. Rủi Ro Và Cách Xử Lý

### Dữ liệu quá nhiều file nhỏ

`data/crops/` có rất nhiều ảnh nhỏ, upload nguyên folder lên Colab sẽ chậm. Cách xử lý:

- Nén thành `crops.zip`.
- Upload một file zip lên Google Drive.
- Giải nén trong Colab.

### Colab train quá lâu

Cách xử lý:

- Bật `USE_SUBSET = True`.
- Train subset cân bằng trước.
- Giảm `EPOCHS`.
- Tăng `BATCH_SIZE` nếu GPU còn đủ VRAM.

### Kết quả baseline chưa cao

Cách xử lý:

- Kiểm tra confusion matrix.
- Kiểm tra các ảnh dự đoán sai.
- Tăng augmentation nhẹ.
- Train lâu hơn.
- Chuyển sang transfer learning.

### Demo chưa có ảnh tự chụp

Cách xử lý:

- Demo bằng hold-out test set.
- Ghi rõ test set không tham gia train.
- Nếu cần demo trực quan hơn, dùng ảnh nguyên đĩa Petri trong raw dataset và bounding box từ XML.

## 10. Kết Luận Định Hướng

Hướng phát triển phù hợp nhất hiện tại là đi theo roadmap nhiều tầng:

```text
Custom CNN crop-classification
-> Transfer learning crop-classification
-> Demo trên test set
-> Demo ảnh nguyên đĩa Petri bằng XML bounding box
-> Object detection
-> Theo dõi nảy mầm theo thời gian
```

Trong phạm vi đồ án hiện tại, baseline Custom CNN đã chứng minh pipeline dữ liệu, mô hình và đánh giá có thể hoạt động. Việc quan trọng tiếp theo là trình bày kết quả rõ ràng, demo được mô hình trên test set và dùng kết quả này làm mốc so sánh cho Transfer Learning.
