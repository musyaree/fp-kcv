"""
Ekstraksi fitur handcrafted (warna) dari citra HSV yang sudah dipreproses.
Fitur ini dipakai sebagai pembanding CNN embedding di Skenario 2.

Cara pakai:
    python src/feature_extraction_handcrafted.py

Output:
    outputs/features_handcrafted_train.npy
    outputs/features_handcrafted_test.npy
"""

from pathlib import Path
import csv
import numpy as np
from scipy.stats import skew

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBSET_DIR = REPO_ROOT / "data" / "processed" / "subset"
PREPROCESSED_DIR = REPO_ROOT / "data" / "processed" / "preprocessed"
MANIFEST_PATH = SUBSET_DIR / "manifest.csv"
OUTPUTS_DIR = REPO_ROOT / "outputs"

def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest tidak ditemukan di {MANIFEST_PATH}. "
            "Jalankan src/data_prep/split_dataset.py terlebih dahulu."
        )
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def hitung_color_moments(row):
    path = PREPROCESSED_DIR / "hsv_normalized" / row["label"] / Path(row["filename"]).with_suffix(".npy")
    arr = np.load(path)  # shape (224, 224, 3) -> channel H, S, V

    fitur = []
    for c in range(3):
        channel = arr[:, :, c]
        fitur.append(channel.mean())
        fitur.append(channel.std())
        fitur.append(skew(channel.flatten()))

    return fitur  # [mean_H, std_H, skew_H, mean_S, std_S, skew_S, mean_V, std_V, skew_V]

def extract_features(rows):
    fitur = []
    for i, row in enumerate(rows):
        fitur.append(hitung_color_moments(row))
        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            print(f"  {i + 1}/{len(rows)} gambar diproses")
    return np.array(fitur)

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_manifest()

    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]

    print(f"Data latih: {len(train_rows)}, Data uji: {len(test_rows)}")

    print("Ekstraksi fitur handcrafted data latih...")
    X_train = extract_features(train_rows)
    print("Ekstraksi fitur handcrafted data uji...")
    X_test = extract_features(test_rows)

    np.save(OUTPUTS_DIR / "features_handcrafted_train.npy", X_train)
    np.save(OUTPUTS_DIR / "features_handcrafted_test.npy", X_test)

    print(f"Selesai. X_train: {X_train.shape}, X_test: {X_test.shape}")

if __name__ == "__main__":
    main()