"""
Ekstraksi fitur CNN dari gambar yang sudah dipreproses (cnn_normalized).
CNN pretrained dipakai cuma buat ekstraksi fitur, tidak dilatih ulang.
Prediksi akhir tetap pakai K-Means, bukan CNN.

Cara pakai:
    python src/feature_extraction.py

Output:
    outputs/features_cnn_train.npy
    outputs/features_cnn_test.npy
    outputs/labels_train.npy
    outputs/labels_test.npy
"""

from pathlib import Path
import csv
import numpy as np
import torch
from torchvision import models

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

def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    return model

def load_array(row):
    path = PREPROCESSED_DIR / "cnn_normalized" / row["label"] / Path(row["filename"]).with_suffix(".npy")
    arr = np.load(path).astype(np.float32)
    arr = arr.transpose(2, 0, 1)  # HWC ke CHW
    return arr

def extract_features(rows, model, batch_size=16):
    fitur = []
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i + batch_size]
        batch_arr = np.stack([load_array(r) for r in batch_rows])
        batch_tensor = torch.from_numpy(batch_arr).float()
        with torch.no_grad():
            out = model(batch_tensor)
        fitur.append(out.numpy())
        print(f"  {i + len(batch_rows)}/{len(rows)} gambar diproses")
    return np.vstack(fitur)

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_manifest()

    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]

    print(f"Data latih: {len(train_rows)}, Data uji: {len(test_rows)}")

    model = load_model()

    print("Ekstraksi fitur data latih...")
    X_train = extract_features(train_rows, model)
    print("Ekstraksi fitur data uji...")
    X_test = extract_features(test_rows, model)

    y_train = np.array([r["label"] for r in train_rows])
    y_test = np.array([r["label"] for r in test_rows])

    np.save(OUTPUTS_DIR / "features_cnn_train.npy", X_train)
    np.save(OUTPUTS_DIR / "features_cnn_test.npy", X_test)
    np.save(OUTPUTS_DIR / "labels_train.npy", y_train)
    np.save(OUTPUTS_DIR / "labels_test.npy", y_test)

    print(f"Selesai. X_train: {X_train.shape}, X_test: {X_test.shape}")

if __name__ == "__main__":
    main()