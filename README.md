# Smart Product Category Assistant — UAS AIB02

**Nama / NIM:** Harun / 38250016
**Mata Kuliah:** Pemrograman Kecerdasan Buatan (AIB02)
**Universitas:** Universitas Bunda Mulia

## Isi Folder

| File | Keterangan |
|---|---|
| `Laporan_UAS_Harun_38250016.pdf` | Laporan lengkap (analisis masalah, dataset, tahapan, hasil & evaluasi, tampilan sistem, kesimpulan) |
| `Project_UAS_Harun_38250016.ipynb` | Notebook utama: load dataset → preprocessing → Elbow/Silhouette → training K-Means → profiling cluster → visualisasi → simpan model → uji prediksi |
| `app.py` | Aplikasi Streamlit "Smart Product Category Assistant" — 5 tab (Analisis Produk, Prediksi Baru, Visualisasi, Evaluasi Model, Dataset) |

## Cara Menjalankan Aplikasi

1. Pastikan Python 3.10+ sudah terpasang.
2. Install dependency:
   ```
   pip install -r requirements.txt
   ```
3. Jalankan notebook `Project_UAS_Harun_38250016.ipynb` terlebih dahulu untuk menghasilkan `models/kmeans_model.pkl`, `models/scaler.pkl`, dan `models/dataset_clustered.csv`.
4. Jalankan aplikasi:
   ```
   streamlit run app.py
   ```

## Ringkasan Model

- Algoritma: K-Means Clustering (K = 3)
- Fitur: Units Sold, Unit Price, Total Revenue
- Silhouette Score: 0.5656
- Label segmen: Premium Stars, Volume Drivers, Standard Items
