# Demo Data

This folder contains a small set of raw Petri-dish images and one short
time-lapse video for the web demo. All demo files are copied or generated from
Faster R-CNN test sequences, so they are not part of the training split.

Use these files to demonstrate the web flow:

```text
Upload demo image
-> Faster R-CNN scratch
-> bounding boxes + germination states
```

The video demo uses the same flow, but the web app samples frames from the MP4
and runs Faster R-CNN on each sampled frame.

## Files

| File | Species | Sequence | Split |
| --- | --- | --- | --- |
| `raw_images/pg1_1_img001.jpg` | Pennisetum glaucum | `pg1_1` | test |
| `raw_images/pg2_12_img025.jpg` | Pennisetum glaucum | `pg2_12` | test |
| `raw_images/sc6_11_img063.jpg` | Secale cereale | `sc6_11` | test |
| `raw_images/zm4_3_img047.jpg` | Zea mays | `zm4_3` | test |
| `videos/pg1_1_timelapse.mp4` | Pennisetum glaucum | `pg1_1` | test |

## Recreate Video

```text
python scripts/create_demo_video.py --sequence-id pg1_1 --output data/demo/videos/pg1_1_timelapse.mp4 --fps 8 --width 960
```
