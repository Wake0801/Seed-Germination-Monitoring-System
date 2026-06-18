# Faster R-CNN Scratch Results

This folder contains the evaluation reports for the improved object-detection
model.

## Model

- Architecture: `fasterrcnn_resnet50_fpn_scratch`
- Pretrained weights: `false`
- Backbone weights: `null`
- Input source: raw images with Pascal VOC XML bounding boxes
- Classes: `background`, `non_germinated`, `germinated`
- Epochs: 30
- Best epoch: 6

## Test Metrics

| Metric | Value |
| --- | ---: |
| mAP | 0.6937 |
| mAP@50 | 0.8973 |
| mAP@75 | 0.8235 |
| mAP non_germinated | 0.7324 |
| mAP germinated | 0.6549 |

## Role In The Project

The Custom CNN baseline is a crop-classification model. Faster R-CNN is the
main improved system model because it works directly on raw images, detects each
seed, and predicts the germination state for every bounding box.

The checkpoint files are kept locally under:

```text
outputs/checkpoints/faster_rcnn_scratch/
```

They are not committed to Git because each `.pth` file is larger than the
normal GitHub file limit.
