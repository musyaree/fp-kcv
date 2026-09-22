"""
Resize gambar untuk jalur CNN, dan
buat versi HSV untuk jalur fitur handcrafted.

Cara pakai:
    python src/preprocess.py

Output:
    data/processed/preprocessed/cnn/<label>/*.jpg   (224x224, RGB)
    data/processed/preprocessed/hsv/<label>/*.jpg   (HSV)
"""

from pathlib import Path
from PIL import Image
import csv
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUBSET_DIR = REPO_ROOT / "data" / "processed" / "subset"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "preprocessed"
TARGET_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # rata-rata per channel R, G, B
IMAGENET_STD = [0.229, 0.224, 0.225]    # standar deviasi per channel R, G, B
MANIFEST_PATH = SUBSET_DIR / "manifest.csv"

def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest tidak ditemukan di {MANIFEST_PATH}. "
            "Jalankan src/data_prep/subset_dataset.py terlebih dahulu."
        )
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def resize_for_cnn(img, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    hasil = img.convert("RGB").resize(TARGET_SIZE)

    hasil.save(dest_path)

def normalize_for_cnn(img, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(img.convert("RGB").resize(TARGET_SIZE))
    arr = arr / 255
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    hasil = (arr - mean) / std

    np.save(dest_path, hasil)

def to_hsv(img, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    hasil = np.array(img.convert("HSV"))

    np.save(dest_path, hasil)

def normalize_hsv(img, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    hasil = np.array(img.convert("HSV"))
    hasil = hasil / 255

    np.save(dest_path, hasil)

def main():
    rows = load_manifest()

    for row in rows:
        src_path = SUBSET_DIR / row["label"] / row["filename"]

        with Image.open(src_path) as img:
            dest_cnn = OUTPUT_DIR / "cnn" / row["label"] / row["filename"]
            resize_for_cnn(img, dest_cnn)

            dest_cnn_norm = OUTPUT_DIR / "cnn_normalized" / row["label"] / Path(row["filename"]).with_suffix(".npy")
            normalize_for_cnn(img, dest_cnn_norm)

            dest_hsv = OUTPUT_DIR / "hsv" / row["label"] / Path(row["filename"]).with_suffix(".npy")
            to_hsv(img, dest_hsv)

            dest_hsv_norm = OUTPUT_DIR / "hsv_normalized" / row["label"] / Path(row["filename"]).with_suffix(".npy")
            normalize_hsv(img, dest_hsv_norm)

    print("Selesai preprocessing.")

if __name__ == "__main__":
    main()