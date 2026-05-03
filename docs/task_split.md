# Team Task Split

This task split is temporary and optimized for the first Deep Learning baseline.

## Phase 1: Simple Baseline

Goal:
- Build a working Deep Learning baseline using cropped seed images.
- Train a simple Custom CNN for binary classification: `non_germinated` vs `germinated`.
- Keep the pipeline reproducible and avoid data leakage by splitting by `sequence_id`.

Person 1: Dataset and Deep Learning input pipeline
- Read Pascal VOC XML files and pair each XML with its JPG image.
- Extract metadata: species, experiment id, dish id, frame id, and `sequence_id`.
- Map raw labels `*_im` and `*_el` to the two baseline classes.
- Crop seed patches from bounding boxes and save them into train/val/test folders.
- Create metadata CSV files: `all_objects.csv`, `sequence_split.csv`, `train.csv`, `val.csv`, `test.csv`.
- Acceptance output: a clean crop dataset that Person 2 and Person 3 can train on without touching raw XML directly.

Person 2: Custom CNN baseline model
- Define a small CNN with 2 to 3 convolution blocks.
- Use only Deep Learning layers: convolution, activation, pooling, dropout, and fully connected layers.
- Prepare image transforms: resize to 224 x 224, tensor conversion, normalization, and light augmentation for training.
- Keep the architecture simple enough to explain in the report.
- Acceptance output: a trainable Custom CNN baseline and a short architecture description.

Person 3: Training, evaluation, and experiment report
- Build the training loop for the Custom CNN baseline.
- Use Cross Entropy Loss and Adam optimizer.
- Track train/val loss and accuracy per epoch.
- Evaluate on the test set using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
- Save checkpoints, logs, plots, and a short baseline result summary.
- Acceptance output: baseline metrics, confusion matrix, loss curve, and selected checkpoint.

## Phase 2: First Improvement After Baseline

Goal:
- Improve the baseline while staying 100 percent Deep Learning.
- Use transfer learning and stronger training strategy, not classical ML.

Person 1: Transfer Learning models
- Implement pretrained CNN backbones for crop classification.
- Candidate models: ResNet18, MobileNetV2, EfficientNet-B0.
- Replace the classification head with a two-class output layer.
- Compare frozen-backbone training and fine-tuning.
- Acceptance output: at least two transfer-learning models trained under the same sequence-based split.

Person 2: Deep Learning training improvements
- Add stronger but biologically reasonable image augmentation.
- Add scheduler support, for example cosine learning-rate schedule.
- Compare Cross Entropy Loss with Focal Loss if class imbalance hurts recall.
- Add early stopping and best-checkpoint selection by validation F1-score.
- Acceptance output: improved training config and controlled comparison against the baseline CNN.

Person 3: Deep Learning explainability and error analysis
- Generate Grad-CAM heatmaps for the best model.
- Analyze false positives and false negatives by species and class.
- Create model comparison tables and figures for the report.
- Prepare the inference interface for later Spring Boot integration.
- Acceptance output: Grad-CAM examples, error-analysis notes, and final model recommendation.

## Explicitly Out Of Scope

- SVM, Random Forest, KNN, Logistic Regression, Naive Bayes, Decision Tree.
- Color thresholding or handcrafted image-processing as the main prediction method.
- HOG, LBP, ORB, SIFT, SURF feature pipelines.
- Any non-Deep-Learning classifier used as the main result.

