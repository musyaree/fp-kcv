"""
EDA (Exploratory Data Analysis) untuk subset dataset DFU.

Membaca manifest.csv hasil subset_dataset.py, lalu:
1. Menghitung jumlah gambar per kategori (sehat vs ulkus).
2. Menampilkan/menyimpan sampel gambar dari tiap kategori.
3. Mencatat statistik ukuran gambar.

Cara pakai:
    python src/eda.py

Output:
    outputs/eda_summary.txt      (jumlah per kategori + statistik ukuran)
    outputs/eda_sample_grid.png  (contoh gambar tiap kategori)
"""

import csv
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUBSET_DIR = REPO_ROOT / "data" / "processed" / "subset"
MANIFEST_PATH = SUBSET_DIR / "manifest.csv"
OUTPUTS_DIR = REPO_ROOT / "outputs"
SAMPLES_PER_CLASS = 5

def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest tidak ditemukan di {MANIFEST_PATH}. "
            "Jalankan src/data_prep/subset_dataset.py terlebih dahulu."
        )
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def summarize_counts(rows):
    return Counter(row["label"] for row in rows)

def summarize_sizes(rows):
    sizes = []
    for row in rows:
        img_path = SUBSET_DIR / row["label"] / row["filename"]
        with Image.open(img_path) as img:
            sizes.append(img.size)
    widths = [w for w, h in sizes]
    heights = [h for w, h in sizes]
    return {
        "n": len(sizes),
        "width_min": min(widths), "width_max": max(widths),
        "height_min": min(heights), "height_max": max(heights),
        "unique_sizes": sorted(set(sizes)),
    }

def save_sample_grid(rows, seed=42):
    random.seed(seed)
    labels = sorted(set(row["label"] for row in rows))

    fig, axes = plt.subplots(len(labels), SAMPLES_PER_CLASS, figsize=(3 * SAMPLES_PER_CLASS, 3 * len(labels)))

    for row_idx, label in enumerate(labels):
        candidates = [r for r in rows if r["label"] == label]
        picks = random.sample(candidates, SAMPLES_PER_CLASS)
        for col_idx, row in enumerate(picks):
            img_path = SUBSET_DIR / row["label"] / row["filename"]
            ax = axes[row_idx, col_idx]
            ax.imshow(Image.open(img_path))
            ax.set_title(row["filename"], fontsize=8)
            ax.axis("off")
        axes[row_idx, 0].set_ylabel(label, fontsize=12)

    fig.suptitle("Sampel Gambar per Kategori")
    fig.tight_layout()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "eda_sample_grid.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def main():
    rows = load_manifest()
    counts = summarize_counts(rows)
    sizes = summarize_sizes(rows)
    grid_path = save_sample_grid(rows)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUTS_DIR / "eda_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== EDA Summary — Subset DFU ===\n\n")
        f.write("Jumlah gambar per kategori:\n")
        for label, count in counts.items():
            f.write(f"  {label}: {count}\n")
        f.write(f"  Total: {sum(counts.values())}\n\n")

        f.write("Statistik ukuran gambar:\n")
        f.write(f"  Width  : min={sizes['width_min']}, max={sizes['width_max']}\n")
        f.write(f"  Height : min={sizes['height_min']}, max={sizes['height_max']}\n")
        f.write(f"  Ukuran unik: {sizes['unique_sizes']}\n\n")

        f.write(f"Sampel visual disimpan di: {grid_path.relative_to(REPO_ROOT)}\n")
        f.write("Sumber dataset: Kaggle Diabetic Foot Ulcer (DFU) Dataset (Laith Jj)\n")
        f.write("https://www.kaggle.com/datasets/laithjj/diabetic-foot-ulcer-dfu\n")

    print(open(summary_path, encoding="utf-8").read())
    print(f"Grid sampel disimpan di: {grid_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
