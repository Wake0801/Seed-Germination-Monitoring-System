# Baseline CNN Results

Thư mục này chứa kết quả train baseline Custom CNN bằng `notebooks/custom_cnn_colab.ipynb`.

## Files

- `best_custom_cnn.pth`: checkpoint tốt nhất theo validation macro F1.
- `training_history.csv`: train/validation loss, accuracy và macro F1 theo epoch.
- `test_summary.json`: metric tổng hợp trên test set.
- `test_metrics.csv`: metric theo từng lớp.
- `learning_curves.png`: biểu đồ loss và macro F1.
- `confusion_matrix.png`: confusion matrix trên test set.

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
