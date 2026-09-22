"""
Membagi manifest.csv menjadi data latih dan data uji (80:20),
per kategori, lalu menambahkan kolom 'split' ke manifest.csv.

Cara pakai:
    python src/split_dataset.py

Output:
    data/processed/subset/manifest.csv   (kolom 'split' ditambahkan: train/test)
"""

from pathlib import Path
import csv
import random

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUBSET_DIR = REPO_ROOT / "data" / "processed" / "subset"
MANIFEST_PATH = SUBSET_DIR / "manifest.csv"
TRAIN_RATIO = 0.8

def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest tidak ditemukan di {MANIFEST_PATH}. "
            "Jalankan src/data_prep/subset_dataset.py terlebih dahulu."
        )
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save_manifest(rows):
    fieldnames = list(rows[0].keys())

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def split_by_label(rows):
    labels = sorted({r["label"] for r in rows})

    for label in labels:
        kelompok = [r for r in rows if r["label"] == label]

        random.shuffle(kelompok)

        n_train = int(len(kelompok) * TRAIN_RATIO)

        train_part = kelompok[:n_train]
        test_part = kelompok[n_train:]

        for row in train_part:
            row["split"] = "train"
        for row in test_part:
            row["split"] = "test"

    return rows

def main():
    random.seed(42)
    rows = load_manifest()
    rows = split_by_label(rows)
    save_manifest(rows)
    print("Selesai split data.")

if __name__ == "__main__":
    main()