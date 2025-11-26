# Analisis dan Prediksi Jumlah Murid Sekolah Menengah Pertama (SMP) di Seluruh Kecamatan Kabupaten Lebak Tahun 2016–2024 Menggunakan Metode Machine Learning

## Deskripsi Proyek

Proyek ini merupakan implementasi dari penelitian yang bertujuan untuk menganalisis dan memprediksi jumlah murid Sekolah Menengah Pertama (SMP) di seluruh kecamatan Kabupaten Lebak dari tahun 2016 hingga 2024 menggunakan metode machine learning. Proyek ini menggabungkan data historis dari berbagai kecamatan untuk membuat prediksi yang akurat dan membantu dalam perencanaan pendidikan di daerah tersebut.

## Tujuan Proyek

- Menganalisis tren jumlah murid SMP di Kabupaten Lebak dari tahun 2016 hingga 2024
- Memprediksi jumlah murid SMP di masa depan menggunakan model machine learning
- Memberikan informasi yang berguna untuk perencanaan pendidikan di Kabupaten Lebak
- Menyediakan visualisasi data untuk memahami pola dan tren lebih baik

## Struktur Proyek

```
prediksi-siswa-lebak/
│
├── dashboard.py                 # Aplikasi dashboard interaktif untuk visualisasi data
├── pre-processing.py            # Skrip untuk membersihkan dan memproses data
├── flowchart.py                 # Alur penelitian visual 
├── requirements.txt             # Daftar dependensi yang dibutuhkan
├── README.md                    # Dokumentasi proyek
├── data_murid_smp_clean.json    # Data bersih hasil dari pre-processing
├── diagram_alur_penelitian.png  # Gambar diagram alur penelitian
└── json/                        # Folder berisi data mentah dari berbagai tahun
    ├── Jumlah Sekolah, Guru, dan Murid Sekolah Menengah Pertama 2016.json
    ├── Jumlah Sekolah, Guru, dan Murid Sekolah Menengah Pertama 2017.json
    ├── Jumlah Sekolah, Guru, dan Murid Sekolah Menengah Pertama 2018.json
    ├── Jumlah Sekolah, Guru, dan Murid Sekolah Menengah Pertama 2019.json
    ├── Jumlah Sekolah, Guru, dan Murid Sekolah Menengah Pertama 2020.json
    ├── Jumlah Sekolah, Guru, dan Murid Sekolah Menengah Pertama 2021.json
    ├── Jumlah Sekolah, Guru, dan Murid Sekolah Menengah Pertama 2022.json
    ├── Jumlah Sekolah, Guru, dan Murid Sekolah Menengah Pertama 2023.json
    └── Jumlah Sekolah, Guru, dan Murid Sekolah Menengah Pertama 2024.json
```

## Daftar Isi

- [Deskripsi Proyek](#deskripsi-proyek)
- [Tujuan Proyek](#tujuan-proyek)
- [Struktur Proyek](#struktur-proyek)
- [Prasyarat](#prasyarat)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Modul dan Fungsi](#modul-dan-fungsi)
- [Dataset](#dataset)
- [Alur Proses](#alur-proses)
- [Visualisasi](#visualisasi)
- [Kesimpulan dan Kontribusi](#kesimpulan-dan-kontribusi)
- [Kontribusi](#kontribusi)
- [Lisensi](#lisensi)

## Prasyarat

Pastikan Anda telah menginstal Python versi 3.7 atau lebih tinggi di sistem Anda.

## Instalasi

1. Clone atau download repositori ini ke lokal komputer Anda:

```bash
git clone <url-repositori>
cd prediksi-siswa-lebak
```

2. Buat environment virtual (disarankan):

```bash
python -m venv venv
```

3. Aktifkan environment virtual:
   - Di Windows:
   ```bash
   venv\Scripts\activate
   ```
   - Di macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. Instal dependensi yang dibutuhkan:

```bash
pip install -r requirements.txt
```

## Penggunaan

### Pre-processing Data

Untuk membersihkan dan menggabungkan data dari berbagai tahun, jalankan:

```bash
python pre-processing.py
```

Skrip ini akan:
- Membaca file JSON dari folder `json/`
- Membersihkan data dari nilai-nilai tidak valid
- Menambahkan kolom tahun ke data
- Menggabungkan semua data ke satu file `data_murid_smp_clean.json`

### Menjalankan Dashboard

Untuk menjalankan dashboard interaktif yang menampilkan visualisasi data:

```bash
streamlit run dashboard.py
```

Dashboard akan tersedia di browser Anda (biasanya di `http://localhost:8501`)

### Flowchart Penelitian

Untuk menampilkan atau memperbarui diagram alur penelitian:

```bash
python flowchart.py
```

## Modul dan Fungsi

### pre-processing.py
- **Fungsi**: Membersihkan dan menggabungkan data dari berbagai tahun
- **Proses**:
  - Membaca file JSON mentah
  - Membersihkan data dari karakter khusus dan nilai tidak valid
  - Menstandarisasi format data
  - Menambahkan informasi tahun ke setiap record
  - Menggabungkan data dari berbagai tahun menjadi satu dataset

### dashboard.py
- **Fungsi**: Menyediakan antarmuka web interaktif untuk visualisasi data
- **Fitur**:
  - Visualisasi tren jumlah murid SMA di berbagai kecamatan
  - Grafik interaktif menggunakan Plotly
  - Filter berdasarkan kecamatan dan tahun
  - Statistik ringkasan

### flowchart.py
- **Fungsi**: Menampilkan diagram alur proses penelitian
- **Output**: Gambar atau tampilan diagram alur penelitian

## Dataset

Dataset yang digunakan dalam proyek ini berisi informasi tentang jumlah sekolah, guru, dan murid Sekolah Menengah Pertama (SMP) di Kabupaten Lebak dari tahun 2016 hingga 2024. Setiap file JSON berisi data per kecamatan dengan informasi:

- Nama kecamatan
- Jumlah sekolah SMP (Negeri, Swasta, dan total)
- Jumlah guru SMP (Negeri, Swasta, dan total) 
- Jumlah murid SMP (Negeri, Swasta, dan total)

Catatan: Meskipun nama file menyebut SMP, proyek ini difokuskan pada analisis dan prediksi jumlah murid SMA.

## Alur Proses

1. **Pengumpulan Data**: Mengumpulkan data dari berbagai tahun dalam format JSON
2. **Pre-processing**: Membersihkan dan menggabungkan data menggunakan `pre-processing.py`
3. **Analisis Data**: Menganalisis tren dan pola dalam data yang telah dibersihkan
4. **Pembangunan Model**: Membangun model machine learning untuk prediksi
5. **Visualisasi**: Membuat visualisasi interaktif menggunakan dashboard
6. **Evaluasi**: Mengevaluasi akurasi model dan hasil prediksi

## Visualisasi

Dashboard yang disediakan mencakup berbagai jenis visualisasi:

- Grafik garis tren jumlah murid SMA
- Grafik perbandingan antar kecamatan
- Grafik distribusi data
- Visualisasi prediksi model

## Kesimpulan dan Kontribusi

Proyek ini berkontribusi pada:

- Pemahaman yang lebih baik tentang tren pendidikan di Kabupaten Lebak
- Alat bantu pengambilan keputusan untuk perencanaan pendidikan
- Implementasi metode machine learning dalam konteks pendidikan
- Dokumentasi dan visualisasi data pendidikan yang dapat diakses

## Kontribusi

Jika Anda ingin berkontribusi pada proyek ini:

1. Fork repositori ini
2. Buat branch fitur (`git checkout -b fitur/fitur-baru`)
3. Commit perubahan Anda (`git commit -m 'Tambah fitur baru'`)
4. Push ke branch (`git push origin fitur/fitur-baru`)
5. Buat pull request

## Lisensi

Proyek ini tersedia secara publik untuk tujuan pendidikan dan penelitian. Jika Anda menggunakan hasil dari proyek ini, harap berikan referensi yang sesuai.

---

Proyek ini merupakan bagian dari penelitian skripsi dalam menganalisis dan memprediksi jumlah murid SMA menggunakan metode machine learning di Kabupaten Lebak.
