# Giới Thiệu Dữ Liệu GermPredDataset

Tài liệu này tóm tắt phần dữ liệu dựa trên hai báo cáo đã đọc trong thư mục gốc của đồ án. Nội dung tập trung vào những thông tin cần thiết để xây dựng baseline Deep Learning.

## Nguồn Dữ Liệu

Dataset sử dụng là GermPredDataset từ Mendeley Data của Genze và Grimm. Dữ liệu phục vụ bài toán nhận diện trạng thái nảy mầm của hạt giống từ ảnh RGB.

Đường dẫn dataset trong dự án:

```text
data/raw/GermPredDataset/
```

## Cấu Trúc Dataset Gốc

Dataset gồm 3 loài hạt:

- `PennisetumGlaucum`: kê ngọc trai.
- `SecaleCereale`: lúa mạch đen.
- `ZeaMays`: ngô.

Mỗi thư mục loài có cùng cấu trúc:

- `img/`: chứa ảnh gốc định dạng `.jpg`.
- `true_ann/`: chứa annotation định dạng `.xml` theo chuẩn Pascal VOC.

Ảnh trong dataset là ảnh một đĩa petri, kích thước 624 x 624, định dạng RGB. Mỗi file XML tương ứng với một ảnh và chứa danh sách bounding box của các hạt xuất hiện trong ảnh.

## Thống Kê Chính

- Tổng số ảnh JPG: 23.797.
- Tổng số file XML: 23.797.
- Tổng số bounding box: 235.933.
- Tổng số chuỗi thời gian: khoảng 247 sequence.
- Mỗi sequence tương ứng với một đĩa petri được chụp liên tiếp theo thời gian.

## Nhãn Dữ Liệu

Nhãn trong XML không ghi trực tiếp là `germinated` hoặc `non_germinated`, mà dùng mã nội bộ theo loài:

- `zm_im`, `sc_im`, `pg_im`: hạt chưa nảy mầm.
- `zm_el`, `sc_el`, `pg_el`: hạt đã nảy mầm.

Với baseline hiện tại, chuẩn hóa về 2 lớp:

```text
*_im -> non_germinated
*_el -> germinated
```

## Quy Tắc Chia Dữ Liệu

Không chia train/val/test ngẫu nhiên theo từng ảnh riêng lẻ. Các frame trong cùng một sequence rất giống nhau, nếu chia sai sẽ làm rò rỉ dữ liệu và khiến kết quả đánh giá bị cao giả tạo.

Quy tắc bắt buộc:

```text
Chia dữ liệu theo sequence_id
```

Ví dụ `sequence_id`:

- `zm1_10`
- `sc3_7`
- `pg2_4`

Toàn bộ ảnh thuộc cùng một `sequence_id` chỉ được nằm trong một tập duy nhất: train, val hoặc test.

## Hướng Sử Dụng Cho Baseline

Baseline đầu tiên dùng hướng phân loại ảnh crop:

1. Parse XML để lấy bounding box.
2. Crop từng hạt từ ảnh gốc.
3. Map nhãn về 2 lớp.
4. Chia dữ liệu theo `sequence_id`.
5. Train Custom CNN trên ảnh crop.
6. Đánh giá kết quả bằng các metric phân loại.

Đây là hướng đơn giản, phù hợp để có kết quả ban đầu trước khi chuyển sang Transfer Learning hoặc các kiến trúc Deep Learning mạnh hơn.
