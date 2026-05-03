# Dữ Liệu Gốc

Thư mục này chứa dataset gốc dùng cho đồ án Deep Learning nhận diện trạng thái nảy mầm hạt giống.

## Đường Dẫn Dataset

```text
data/raw/GermPredDataset/
```

## Nguồn Dữ Liệu

Dataset là GermPredDataset từ Mendeley Data của Genze và Grimm. Dữ liệu gồm ảnh RGB của các đĩa petri và annotation vị trí từng hạt giống.

## Cấu Trúc

```text
GermPredDataset/
+-- PennisetumGlaucum/
|   +-- img/
|   +-- true_ann/
+-- SecaleCereale/
|   +-- img/
|   +-- true_ann/
+-- ZeaMays/
    +-- img/
    +-- true_ann/
```

Ý nghĩa:

- `img/`: ảnh gốc định dạng `.jpg`.
- `true_ann/`: annotation định dạng `.xml` theo chuẩn Pascal VOC.

## Nhãn

Các nhãn gốc trong XML gồm:

- `pg_im`, `sc_im`, `zm_im`: chưa nảy mầm.
- `pg_el`, `sc_el`, `zm_el`: đã nảy mầm.

Với baseline hiện tại, nhãn được map về 2 lớp:

```text
*_im -> non_germinated
*_el -> germinated
```

## Lưu Ý

Không chỉnh sửa trực tiếp dữ liệu trong thư mục này. Các file crop, metadata và tập train/val/test sẽ được sinh ra ở các thư mục khác.

Raw dataset đang bị ignore khỏi Git vì số lượng file lớn. Khi push GitHub, chỉ file README này được commit.
