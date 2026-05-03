# Dataset Notes From Reports

Source reports read:
- Bao cao Xu ly Du lieu GermPredDataset.docx
- Bao cao Deep Learning cho He thong Giam sat Nay mam.docx

Dataset summary:
- Source: GermPredDataset from Mendeley Data by Genze and Grimm.
- Species: `ZeaMays`, `SecaleCereale`, `PennisetumGlaucum`.
- Local structure per species: `img/` for JPG images and `true_ann/` for Pascal VOC XML annotations.
- Reported total: 23,797 JPG images, 23,797 XML files, 235,933 bounding boxes.
- Image size: 624 x 624 RGB.
- Time-series groups: 247 sequences, each sequence represents one petri dish over time.

Label mapping for the current baseline:
- `zm_im`, `sc_im`, `pg_im` -> `non_germinated`
- `zm_el`, `sc_el`, `pg_el` -> `germinated`

Critical data rule:
- Split by `sequence_id`, for example `zm1_10`, `sc3_7`, `pg2_4`.
- Do not split randomly at image level because adjacent frames in the same sequence are visually similar and would cause data leakage.

Baseline direction:
- Crop each seed object from XML bounding boxes.
- Train a small Custom CNN on cropped seed images.
- Evaluate on a held-out test set split by `sequence_id`.

Strict method boundary:
- This project must focus on Deep Learning models.
- Classical ML models and handcrafted computer-vision methods are not part of the main method.

