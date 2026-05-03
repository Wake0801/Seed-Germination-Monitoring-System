# Huấn Luyện Và Đánh Giá

Thư mục này chứa các script huấn luyện và đánh giá mô hình Deep Learning.

## File Chính

- `train_baseline.py`: train Custom CNN baseline trên dữ liệu crop.
- `evaluate.py`: đánh giá checkpoint trên tập test và xuất metric, biểu đồ, báo cáo.

## Metric Baseline

Các metric cần có:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion Matrix.

## Output

Kết quả huấn luyện được lưu vào:

```text
outputs/checkpoints/
outputs/logs/
outputs/reports/
outputs/figures/
```

## Lưu Ý

Kết quả baseline phải được đánh giá trên test set chia theo `sequence_id`. Không dùng kết quả từ tập train để báo cáo hiệu năng mô hình.
