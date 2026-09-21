"""
Mengunduh Kaggle Diabetic Foot Ulcer (DFU) Dataset dan menyalinnya
ke folder data/raw/ pada repo ini.

Cara pakai:
    python download_data.py

Persyaratan:
    pip install kagglehub
"""

import shutil
from pathlib import Path

import kagglehub

DATASET_SLUG = "laithjj/diabetic-foot-ulcer-dfu"
TARGET_DIR = Path(__file__).resolve().parent / "data" / "raw"


def main():
    print(f"Mengunduh dataset '{DATASET_SLUG}' ...")
    cache_path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"Dataset berhasil diunduh ke cache: {cache_path}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = Path(cache_path)
    for item in cache_path.iterdir():
        dest = TARGET_DIR / item.name
        if dest.exists():
            print(f"  Lewati (sudah ada): {dest}")
            continue
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
        print(f"  Disalin ke: {dest}")

    print(f"\nSelesai. Dataset tersedia di: {TARGET_DIR}")


if __name__ == "__main__":
    main()
