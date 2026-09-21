# Pengelompokan Tingkat Keparahan Luka Diabetes dari Citra Kaki Menggunakan CNN dan K-Means

## Deskripsi Singkat

Proyek ini mengelompokkan citra kaki penderita diabetes berdasarkan
kemiripan fitur visual, dengan tujuan mengeksplorasi apakah pola yang terbentuk
secara alami dapat diinterpretasikan sebagai tingkat keparahan luka. Fitur citra 
diekstraksi menggunakan CNN pretrained, sedangkan prediksi 
akhir/pengelompokan dilakukan dengan algoritma K-Means.

## Anggota Kelompok

| Nama | NRP |
|------|-----|
| FAEYZAR AHNAF MUSYARRI | 5025251117 |
| EVINA FITRIYANI | 5054251013 |
| RAIHAN NAUFAL RAMADHAN | 5025251193 |
| KOMANG MAHATMA LANGENDRIA | 5025251151 |

## Sumber Dataset

Kaggle Diabetic Foot Ulcer (DFU) Dataset — Laith Jj
https://www.kaggle.com/datasets/laithjj/diabetic-foot-ulcer-dfu

## Struktur Folder

```
.
├── data/
│   └── raw/              # dataset mentah (terisi otomatis via download_data.py)
├── notebook/             # notebook utama (.ipynb)
├── src/                  # fungsi pendukung (ekstraksi fitur, clustering, dll)
├── outputs/              # hasil eksperimen: fitur (.npy), grafik, visualisasi cluster
├── download_data.py      # script untuk mengunduh dataset ke data/raw/
├── requirements.txt
└── README.md
```

## Cara Menjalankan

1. Clone repository ini:
   ```bash
   git clone <url-repo-ini>
   cd <nama-repo>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download dataset:
   ```bash
   python download_data.py
   ```

4. Jalankan notebook utama:
   ```bash
   jupyter notebook notebook/
   ```
   Buka file notebook, lalu jalankan seluruh cell secara berurutan (Run All).

## Metode

1. **Pra-pemrosesan citra**: resize, normalisasi, split data latih/uji.
2. **Ekstraksi fitur (Deep Learning)**: embedding CNN pretrained + fitur handcrafted (warna) sebagai pembanding.
3. **Reduksi dimensi**: PCA.
4. **Clustering**: K-Means.
5. **Evaluasi**: Silhouette Score, interpretasi visual & domain (referensi skala keparahan medis).

## Skenario Eksperimen

| Skenario | Yang Diuji |
|----------|------------|
| 1 | Jumlah cluster (K = 2, 3, 4) |
| 2 | Jenis fitur (CNN embedding vs handcrafted) |
| 3 | Dengan vs tanpa PCA |
