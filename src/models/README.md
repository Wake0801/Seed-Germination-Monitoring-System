# Mô Hình Deep Learning

Thư mục này chứa định nghĩa các mô hình Deep Learning dùng trong đồ án.

## Baseline

- `custom_cnn.py`: Custom CNN đơn giản cho phân loại ảnh crop hạt giống thành 2 lớp.

Baseline nên có kiến trúc dễ giải thích, ví dụ 2 đến 3 block convolution, sau đó là fully connected layer.

## Cải Tiến

- `transfer_learning.py`: các mô hình Transfer Learning như ResNet18, MobileNetV2 hoặc EfficientNet-B0.

Các mô hình cải tiến chỉ được thực hiện sau khi baseline Custom CNN chạy ổn định.

## Nguyên Tắc

- Chỉ dùng mô hình Deep Learning.
- Không đặt SVM, Random Forest, KNN hoặc các mô hình Machine Learning cổ điển trong thư mục này.
- Mỗi model cần có mô tả rõ input size, số lớp đầu ra và mục đích sử dụng.
