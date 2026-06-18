# Hướng Phát Triển Đồ Án

Tài liệu này mô tả hướng phát triển hiện tại của đồ án giám sát nảy mầm hạt giống bằng Deep Learning sau khi có kết quả baseline Custom CNN và model cải tiến Faster R-CNN train from scratch.

## 1. Định Hướng Chính

Hướng triển khai được chốt:

```text
Custom CNN baseline trên ảnh crop
-> Faster R-CNN scratch trên ảnh raw + XML bounding box
-> Web dashboard upload ảnh/video, hiển thị bounding box và trạng thái hạt
```

Lý do chọn hướng này:

- Không dùng pretrained model, phù hợp ràng buộc hiện tại.
- Baseline Custom CNN chứng minh pipeline crop classification hoạt động.
- Faster R-CNN scratch giải quyết trực tiếp bài toán hệ thống: phát hiện từng hạt và phân loại trạng thái trên ảnh nguyên đĩa Petri.
- Web demo có thể hiển thị bounding box + trạng thái ngay trên ảnh, trực quan hơn so với chỉ phân loại một ảnh crop.

## 2. Baseline Custom CNN

Baseline hiện tại là mô hình phân loại ảnh crop thành 2 lớp:

```text
germinated
non_germinated
```

Kết quả test:

| Metric | Giá trị |
| --- | ---: |
| Accuracy | 0.8336 |
| Macro Precision | 0.8327 |
| Macro Recall | 0.8308 |
| Macro F1-score | 0.8316 |
| Weighted F1-score | 0.8334 |

Vai trò trong báo cáo:

- Là mốc so sánh đầu tiên.
- Chứng minh bước parse XML, crop ảnh, split dữ liệu và train model đã chạy được.
- Dùng để giải thích hạn chế của crop classification: chỉ dự đoán được khi đã có sẵn crop từng hạt.

## 3. Model Cải Tiến: Faster R-CNN Scratch

Model cải tiến hiện tại:

```text
fasterrcnn_resnet50_fpn_scratch
```

Đặc điểm:

- Train từ đầu, không dùng pretrained weights.
- Input là ảnh raw.
- Annotation dùng Pascal VOC XML bounding box.
- Output gồm bounding box, class và confidence cho từng hạt.
- Class gồm `background`, `non_germinated`, `germinated`.

Kết quả test:

| Metric | Giá trị |
| --- | ---: |
| mAP | 0.6937 |
| mAP@50 | 0.8973 |
| mAP@75 | 0.8235 |
| mAP non_germinated | 0.7324 |
| mAP germinated | 0.6549 |

Nhận xét:

- `mAP@50 = 0.8973` đủ tốt để demo hệ thống phát hiện hạt trên ảnh raw.
- Lớp `non_germinated` đạt mAP cao hơn `germinated`, phù hợp với quan sát rằng hạt đã nảy mầm có hình dạng đa dạng hơn.
- Best checkpoint ở epoch 6, cho thấy model đạt đỉnh sớm; báo cáo nên dùng checkpoint tốt nhất thay vì epoch cuối.

## 4. Hệ Thống Web

Web dashboard gồm 2 phần:

- `Monitoring System`: upload ảnh raw, gọi Faster R-CNN, vẽ bounding box và thống kê số hạt theo trạng thái.
- `Model Comparison`: so sánh baseline Custom CNN với Faster R-CNN theo vai trò và metric phù hợp.

Luồng hệ thống:

```text
Upload image
-> FastAPI backend
-> Faster R-CNN scratch checkpoint
-> JSON detections
-> Frontend vẽ box + label + confidence
```

Checkpoint `.pth` chạy bằng Python/PyTorch backend. Không cần ONNX trừ khi muốn chạy inference trực tiếp trong browser hoặc deploy bằng ONNX Runtime.

## 5. Phần Nên Trình Bày Trong Báo Cáo

Nên trình bày theo câu chuyện kỹ thuật:

1. Tiền xử lý dữ liệu từ ảnh raw và XML.
2. Tạo crop dataset để train baseline Custom CNN.
3. Đánh giá baseline bằng Accuracy, Precision, Recall, F1-score.
4. Nêu hạn chế của baseline: cần crop sẵn từng hạt.
5. Cải tiến bằng Faster R-CNN scratch để phát hiện và phân loại trực tiếp trên ảnh raw.
6. Đánh giá Faster R-CNN bằng mAP, mAP@50, mAP@75.
7. Demo web upload ảnh và hiển thị bounding box + trạng thái.

## 6. Hướng Mở Rộng

Sau phiên bản hiện tại, có thể mở rộng:

- Xử lý video theo frame và lưu kết quả từng frame.
- Tracking seed ID qua thời gian để theo dõi tiến trình nảy mầm.
- Xuất file CSV/JSON thống kê theo frame.
- Thử các kiến trúc detection khác nhưng vẫn train from scratch nếu ràng buộc không cho dùng pretrained.
