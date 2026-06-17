# Baseline CNN Results

Thư mục này chứa kết quả train baseline Custom CNN bằng `notebooks/custom_cnn_colab.ipynb`.

## Files

- `outputs/checkpoints/baseline_cnn/best_custom_cnn.pth`: checkpoint tốt nhất theo validation macro F1.
- `outputs/logs/baseline_cnn/training_history.csv`: train/validation loss, accuracy và macro F1 theo epoch.
- `outputs/reports/baseline_cnn/test_summary.json`: metric tổng hợp trên test set.
- `outputs/reports/baseline_cnn/test_metrics.csv`: metric theo từng lớp.
- `outputs/figures/baseline_cnn/learning_curves.png`: biểu đồ loss và macro F1.
- `outputs/figures/baseline_cnn/confusion_matrix.png`: confusion matrix trên test set.

## Test Summary

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8336 |
| Macro Precision | 0.8327 |
| Macro Recall | 0.8308 |
| Macro F1-score | 0.8316 |
| Weighted F1-score | 0.8334 |

## Notes

Baseline được train bằng Colab notebook thay vì script local vì máy local không phù hợp để train toàn bộ crop dataset. Các script `src/training/train_baseline.py` và `src/training/evaluate.py` hiện không phải pipeline chính.
