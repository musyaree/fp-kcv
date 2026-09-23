"""
Menghilangkan gambar yang duplikat dan mengambil subset seimbang dari Patches/Abnormal(Ulcer) dan Patches/Normal(Healthy skin)
di dalam dataset DFU mentah, lalu menyalinnya ke data/processed/subset/.

Cara pakai:
    python src/data_prep/subset_dataset.py
    python src/data_prep/subset_dataset.py --per-class 150

Output:
    data/processed/subset/Abnormal/*.jpg
    data/processed/subset/Normal/*.jpg
    data/processed/subset/manifest.csv
"""

import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PATCHES_DIR = REPO_ROOT / "data" / "raw" / "DFU" / "Patches"
CLASS_FOLDERS = {
    "Abnormal": "Abnormal(Ulcer)",
    "Normal": "Normal(Healthy skin)",
}

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-class", type=int, default=240,
        help="Jumlah gambar yang diambil per kelas",
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "data" / "processed" / "subset",
        help="Folder tujuan hasil subset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed agar hasil reproducible.")
    return parser.parse_args(argv)

def remove_duplicates(files):
    seen = set()
    unique = []
    for path in files:
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(path)
    return unique

def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)

    if not PATCHES_DIR.exists():
        raise FileNotFoundError(
            f"Folder Patches tidak ditemukan di {PATCHES_DIR}. "
            "Jalankan download_data.py terlebih dahulu."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for label, folder_name in CLASS_FOLDERS.items():
        source_dir = PATCHES_DIR / folder_name
        all_files = sorted(p for p in source_dir.iterdir() if p.is_file())
        files = remove_duplicates(all_files)
        print(f"{label}: {len(all_files)} file, {len(files)} unik "
              f"({len(all_files) - len(files)} duplikat dibuang)")

        if len(files) < args.per_class:
            raise ValueError(
                f"Hanya ada {len(files)} gambar unik di '{folder_name}', "
                f"tidak cukup untuk --per-class {args.per_class}."
            )

        selected = random.sample(files, args.per_class)

        dest_dir = args.output / label
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        for src_path in selected:
            dest_path = dest_dir / src_path.name
            shutil.copy2(src_path, dest_path)
            manifest_rows.append({
                "filename": src_path.name,
                "label": label,
                "source_path": str(src_path.relative_to(REPO_ROOT)),
            })

        print(f"{label}: {len(selected)} gambar disalin ke {dest_dir.relative_to(REPO_ROOT)}")

    manifest_path = args.output / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "source_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nTotal: {len(manifest_rows)} gambar. Manifest disimpan di {manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
