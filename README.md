# Seed Germination Deep Learning

Temporary project skeleton for the seed germination monitoring Deep Learning work.

Scope for the first baseline:
- Use only Deep Learning models.
- Build a simple binary classifier on cropped seed images.
- Labels: `*_im` -> `non_germinated`, `*_el` -> `germinated`.
- Split train/val/test by `sequence_id`, not by random image rows.
- Do not use classical Machine Learning models such as SVM, Random Forest, KNN, Logistic Regression, handcrafted thresholding, HOG, or LBP as the main method.

Expected local dataset source:
- `../GermPredDataset/GermPredDataset/`

Main folders:
- `data/`: processed crop dataset and metadata generated from the raw GermPredDataset.
- `src/data/`: XML parsing, crop generation, label mapping, and sequence-based split.
- `src/models/`: Deep Learning model definitions.
- `src/training/`: training and evaluation entrypoints.
- `src/inference/`: prediction entrypoint for later system integration.
- `outputs/`: checkpoints, logs, reports, and figures.
- `docs/`: project notes and team task split.

