import pandas as pd
import glob
import os
import re
import numpy as np
import json

def load_and_clean_json_data():
    """
    Fungsi utama untuk memuat dan membersihkan data dari file-file JSON
    """
    print("Memulai proses penggabungan data dari file JSON...")

    # Tentukan path folder JSON
    folder_path = './json'
    file_pattern = "Jumlah Sekolah, Guru, dan Murid*.json"
    search_path = os.path.join(folder_path, file_pattern)

    # Cari semua file JSON yang cocok dengan pola
    json_files = glob.glob(search_path)
    json_files.sort()

    # Validasi apakah ada file yang ditemukan
    if not json_files:
        print(f"❌ Error: Tidak ada file JSON yang cocok dengan pola '{file_pattern}' ditemukan di '{folder_path}'.")
        return pd.DataFrame()  # Kembalikan DataFrame kosong
    else:
        print(f"✅ Ditemukan {len(json_files)} file JSON:")
        for f in json_files:
            print(f"   - {os.path.basename(f)}")

    # Inisialisasi variabel untuk menyimpan data
    all_dataframes = []
    failed_files = []

    # Proses setiap file JSON
    print("\nMemulai pemrosesan setiap file JSON...")
    for file_path in json_files:
        filename = os.path.basename(file_path)
        print(f"\n🔄 Memproses file: {filename}...")

        try:
            # 1. Ekstrak tahun dari nama file
            match = re.search(r'(\d{4})', filename)
            if not match:
                print(f"   ⚠️ Peringatan: Tidak dapat mengekstrak tahun dari '{filename}'. Melewati.")
                failed_files.append(filename + " (Tahun tidak terdeteksi)")
                continue

            year = int(match.group(1))
            print(f"   - Tahun terdeteksi: {year}")

            # 2. Baca file JSON secara manual
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f) # Ini akan memuat file sebagai list Python

            # 3. Filter list: Hapus item 'null' (yang menjadi 'None' di Python)
            # Kita hanya ambil item yang merupakan 'dict' (object data)
            cleaned_list = [item for item in raw_data if isinstance(item, dict)]

            # 4. Buat DataFrame dari list yang sudah bersih
            df = pd.DataFrame(cleaned_list)

            # Validasi: Jika DataFrame kosong setelah filter (misal file hanya berisi null)
            if df.empty:
                print(f"   ⚠️ Peringatan: Tidak ada data valid (dictionary) di '{filename}'. Melewati.")
                failed_files.append(filename + " (Kosong setelah filter null)")
                continue

            # 5. Langkah Pembersihan (diterapkan pada df baru)

            # 5a. Ganti placeholder '-' dengan NaN
            df.replace('-', np.nan, inplace=True)

            # 5b. Hapus baris ringkasan/total/catatan
            keywords_to_remove = ['Lebak', 'Jumlah', 'Total', 'Catatan', 'sup>']
            pattern = '|'.join(keywords_to_remove)
            df_cleaned = df[~df['Kecamatan'].astype(str).str.contains(pattern, case=False, na=False)].copy()

            # 5c. Tambahkan kolom 'Tahun'
            df_cleaned['Tahun'] = year

            # 5d. Konversi semua kolom data ke numerik
            cols_to_convert = [col for col in df_cleaned.columns if col not in ['Kecamatan', 'Tahun']]
            for col in cols_to_convert:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

            # Tambahkan DataFrame ke list
            all_dataframes.append(df_cleaned)
            print(f"   ✅ Berhasil diproses. Baris bersih: {len(df_cleaned)}")

        except Exception as e:
            print(f"   ❌ Error saat memproses file {filename}: {e}. Melewati file ini.")
            failed_files.append(filename + f" (Error: {e})")

    # Gabungkan semua DataFrame menjadi satu dataset besar
    print("\n-------------------------------------")
    if not all_dataframes:
        print("❌ Tidak ada data yang berhasil diproses. DataFrame final kosong.")
        return pd.DataFrame()
    else:
        print(f"✅ Berhasil memproses {len(all_dataframes)} dari {len(json_files)} file.")
        if failed_files:
            print("\n⚠️ File yang gagal diproses atau dilewati:")
            for failed in failed_files:
                print(f"   - {failed}")

        print(f"\n🌀 Menggabungkan {len(all_dataframes)} dataset...")
        final_df = pd.concat(all_dataframes, ignore_index=True)
        print("   ✅ Penggabungan selesai.")

        # Simpan dataset hasil ke file JSON
        output_filename = "data_murid_smp_clean.json"
        print(f"\n💾 Menyimpan dataset bersih ke file '{output_filename}'...")
        final_df.to_json(output_filename, orient='records', indent=4, force_ascii=False)

        # Tampilkan informasi dataset gabungan
        print("\n--- Info Dataset Gabungan Final ---")
        final_df.info()

        print("\n--- 5 Baris Pertama Data Bersih ---")
        print(final_df.head())

        print(f"\n🚀 Proses selesai. Data bersih Anda kini tersedia dalam variabel `final_df` dan file `{output_filename}`.")

        return final_df

# Panggil fungsi utama
if __name__ == "__main__":
    final_df = load_and_clean_json_data()