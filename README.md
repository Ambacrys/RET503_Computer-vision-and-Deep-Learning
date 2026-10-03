# Klasifikasi Sendok vs Garpu — Transfer Learning EfficientNetB0

Mata kuliah **computer vision and deep learning* ·  RE503
 · Nama: **[Crystiade Eka Syahputra Tarigan – 4222401024]**

Proyek ini membandingkan 3 mode pelatihan EfficientNetB0 (*scratch*, *feature extraction*,
*fine-tuning*) untuk klasifikasi biner **fork vs spoon**, lalu mengukur latensi inferensi model terpilih.

## Struktur repo

```
.
├── README.md
├── metadata.csv              # daftar semua citra + label + split
├── dataset_raw/              # citra mentah per kelas
│   ├── fork/                 # 199 citra
│   └── spoon/                # 151 citra
├── docs/desain_awal.md       # dokumen desain awal (per kelompok)
├── src/
│   ├── make_metadata.py      # buat metadata.csv + split
│   ├── train.py              # training 3 mode
│   ├── plot_results.py       # grafik akurasi per epoch + tabel hasil
│   └── measure_latency.py    # latensi model terpilih
├── results/                  # history, metrics, grafik, tabel, latensi
├── models/                   # model .keras hasil training
└── requirements.txt
```

## Dataset

- Sumber: dataset **spoon-vs-fork** (`archive.zip`) — 
- Jumlah citra: **fork = 199**, **spoon = 151**, total **350** (syarat ≥ 50 per kelas terpenuhi)
- Split stratified, seed 42, rasio 70/15/15 (kolom `split` di `metadata.csv`):

| Split | fork | spoon | Total |
|---|---|---|---|
| train | 139 | 105 | 244 |
| validation | 30 | 23 | 53 |
| test | 30 | 23 | 53 |

### Ringkasan & kualitas data

| Aspek | Temuan |
|---|---|
| Keseimbangan kelas | fork 56,9% : spoon 43,1% (cukup seimbang; macro-F1 ikut dilaporkan) |
| Resolusi | fork 180–1000 px (median lebar 500); spoon 224–500 px (median 500) |
| Format (dari isi file) | JPEG 320, PNG 28, WEBP 2; semua 350 file terbaca, tidak ada yang rusak |
| Ekstensi janggal | 3 file fork berekstensi `.php`/`.ashx` tetapi isinya JPEG asli; dibaca berdasarkan isi file, jadi tidak bermasalah |
| Duplikat | folder arsip asli berisi salinan bersarang `spoon-vs-fork/spoon-vs-fork/` yang identik (dicek MD5) → **dibuang**; tidak ada duplikat persis antar file |
| Near-duplicate | 2 pasangan gambar sangat mirip (1 di fork, 1 di spoon) berpotensi jatuh di split berbeda; dampaknya kecil (2 dari 350 gambar) |

`metadata.csv` berisi: `filepath, label, split, format, width, height, size_kb`.

## Cara menjalankan

**Google Colab** (TensorFlow sudah tersedia):

```python
!git clone https://github.com/<username>/<nama-repo>.git
%cd <nama-repo>
!pip install -q pandas pillow

!python src/make_metadata.py   # opsional: metadata.csv sudah ada di repo
!python src/train.py --mode scratch
!python src/train.py --mode feature_extraction
!python src/train.py --mode fine_tuning
!python src/plot_results.py
!python src/measure_latency.py
```

**Lokal:** `pip install -r requirements.txt`, lalu jalankan perintah `python src/...` yang sama.

## Hasil 3 mode

> Salin isi `results/results_table.md` ke sini setelah training selesai.

| Mode | Epoch | Best val acc | Best val loss | Test acc | Test macro-F1 | Test loss | Param trainable | Waktu training (s) |
|---|---|---|---|---|---|---|---|---|
| Scratch | – | – | – | – | – | – | – | – |
| Feature extraction | – | – | – | – | – | – | – | – |
| Fine-tuning | – | – | – | – | – | – | – | – |

### Grafik akurasi per epoch

![Akurasi per epoch](results/accuracy_per_epoch.png)

![Perbandingan akurasi validasi](results/accuracy_val_comparison.png)

![Confusion matrix (test set)](results/confusion_matrices.png)

## Latensi model terpilih

Model terpilih: **[Feature]** (dipilih dari akurasi validasi). Pengukuran: batch = 1, input 224×224×3,
20 warm-up + 100 run, hanya forward pass (tanpa baca file/resize).

> Salin isi `results/latency.md` ke sini.

| Model | Device | Mean (ms) | Median (ms) | P95 (ms) | FPS | Ukuran model (MB) |
|---|---|---|---|---|---|---|
| – | – | – | – | – | – | – |

## Analisis singkat

1. **Perbandingan mode.** Mode ___ memberi akurasi validasi tertinggi (__%), sedangkan *scratch* hanya __%.
   Ini menunjukkan bahwa [pretrained ImageNet membantu / tidak membantu] pada dataset sekecil ini karena ___.
2. **Kurva training.** [Jelaskan: apakah ada overfitting (train acc jauh di atas val acc)? kapan EarlyStopping berhenti? apakah fine-tuning menaikkan atau menurunkan val acc setelah unfreeze?]
3. **Latensi.** Model terpilih berjalan ± __ ms/gambar di [CPU/GPU] (≈ __ FPS), ukuran __ MB.
   [Cukup/tidak cukup untuk aplikasi real-time karena ___.]
4. **Keterbatasan.** Validation dan test masing-masing hanya 53 citra, sehingga satu gambar salah
   mengubah akurasi sekitar 1,9 poin persen. Ada 2 pasangan near-duplicate dan sebagian gambar berlatar
   putih ala foto produk, jadi performa pada foto dunia nyata bisa lebih rendah dari angka di atas.

## Checklist keluaran P2

| Keluaran | Lokasi |
|---|---|
| Dokumen desain awal (per kelompok) | `docs/desain_awal.md` |
| `dataset_raw` + `metadata.csv`, ≥ 50 citra/kelas | `dataset_raw/`, `metadata.csv` |
| Tabel hasil 3 mode + grafik akurasi per epoch | bagian *Hasil 3 mode*, `results/` |
| Latensi model yang dipilih | bagian *Latensi*, `results/latency.md` |
| Analisis singkat | bagian *Analisis singkat* |
