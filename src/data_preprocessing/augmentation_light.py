# augmentation_light.py
"""
Light augmentation configuration for seed germination YOLO training.

Dataset:
- Object detection
- 2 classes:
    0: non-germinated
    1: germinated
- Images resized to 640x640

Mục tiêu:
- Augmentation nhẹ
- Không làm biến dạng mạnh vùng hạt, mầm, rễ
- Không dùng mosaic/mixup/copy-paste trong cấu hình chính
"""

from pathlib import Path
import yaml


AUGMENTATION_LIGHT = {
    # HSV color augmentation
    # Thay đổi màu/ánh sáng nhẹ
    "hsv_h": 0.005,
    "hsv_s": 0.25,
    "hsv_v": 0.15,

    # Geometric augmentation
    # Xoay/dịch/scale nhẹ
    "degrees": 10.0,
    "translate": 0.05,
    "scale": 0.10,

    # Tắt biến dạng mạnh
    "shear": 0.0,
    "perspective": 0.0,

    # Flip hợp lý vì đĩa petri không có hướng cố định
    "flipud": 0.5,
    "fliplr": 0.5,

    # Tắt augmentation mạnh trong cấu hình chính
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}


def save_yaml(output_path: str = "augmentation_light.yaml") -> None:
    output_path = Path(output_path)

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            AUGMENTATION_LIGHT,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"Saved augmentation config to: {output_path.resolve()}")


def save_cli_args(output_path: str = "augmentation_light_args.txt") -> None:
    output_path = Path(output_path)

    args = " ".join(
        f"{key}={value}"
        for key, value in AUGMENTATION_LIGHT.items()
    )

    with output_path.open("w", encoding="utf-8") as f:
        f.write(args + "\n")

    print(f"Saved YOLO CLI augmentation args to: {output_path.resolve()}")


def print_config() -> None:
    print("===== LIGHT AUGMENTATION CONFIG =====")

    for key, value in AUGMENTATION_LIGHT.items():
        print(f"{key}: {value}")

    print("\nYOLO CLI args:")
    print(
        " ".join(
            f"{key}={value}"
            for key, value in AUGMENTATION_LIGHT.items()
        )
    )


if __name__ == "__main__":
    print_config()
    save_yaml()
    save_cli_args()