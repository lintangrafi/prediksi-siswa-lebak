import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_percentage_error
import os

# --- 1. Konfigurasi Halaman dan Judul ---
st.set_page_config(layout="wide", page_title="Prediksi Murid SMP Lebak")
st.title("Dashboard Prediksi Jumlah Murid SMP di Kabupaten Lebak")

# --- 2. Fungsi Pemuatan Data (dengan Fitur Lag) ---
@st.cache_data
def load_data(file_path="data_murid_smp_clean.json"):
    if not os.path.exists(file_path):
        st.error(f"ERROR: File '{file_path}' tidak ditemukan. Pastikan file berada di folder yang sama.")
        return pd.DataFrame()
    
    df = pd.read_json(file_path)
    if df.empty:
        st.error("Data berhasil dimuat tetapi kosong.")
        return pd.DataFrame()
        
    df = df.sort_values(by=['Kecamatan', 'Tahun'])
    df['Jumlah Murid Tahun Lalu'] = df.groupby('Kecamatan')['Jumlah Murid SMP (Negeri+Swasta)'].shift(1)
    return df

# --- 3. Fungsi Pelatihan Model (dengan Fitur Lag) ---
@st.cache_resource
def train_model(df_input):
    target_col = 'Jumlah Murid SMP (Negeri+Swasta)'
    df_clean = df_input.dropna(subset=[target_col]).copy()
    features_to_use = [
        'Kecamatan', 'Tahun', 'Jumlah Sekolah SMP (Negeri)', 'Jumlah Sekolah SMP (Swasta)',
        'Jumlah Sekolah SMP (Negeri+Swasta)', 'Jumlah Guru SMP (Negeri)',
        'Jumlah Guru SMP (Swasta)', 'Jumlah Guru SMP (Negeri+Swasta)',
        'Jumlah Murid Tahun Lalu'
    ]
    X = df_clean[features_to_use]
    y = df_clean[target_col]
    categorical_features = ['Kecamatan']
    numeric_features = [col for col in features_to_use if col not in categorical_features]
    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])
    model_rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=10, random_state=42, n_jobs=-1)
    model_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', model_rf)])
    model_pipeline.fit(X, y)
    return model_pipeline, features_to_use

# --- 4. Muat Data dan Latih Model ---
df = load_data()

if not df.empty:
    df_for_training = df[df['Tahun'] > 2016].copy() 
    model, features = train_model(df_for_training)
    
    # --- 5. Tampilan Sidebar (Filter) ---
    st.sidebar.header("Filter Tampilan")
    kecamatan_list = ["Semua Kecamatan"] + sorted(df['Kecamatan'].unique())
    selected_kecamatan = st.sidebar.selectbox(
        "Pilih Kecamatan:",
        options=kecamatan_list,
        help="Pilih 'Semua Kecamatan' untuk melihat akurasi model global, atau kecamatan spesifik untuk melihat tren."
    )

    # --- 6. Tampilan Utama (Dashboard) ---
    
    if selected_kecamatan == "Semua Kecamatan":
        st.header("Akurasi Model Global (Data Uji 2024)")
        
        df_test = df[df['Tahun'] == 2024].copy()
        
        if not df_test.empty and 'Jumlah Murid Tahun Lalu' in df_test.columns:
            y_test = df_test['Jumlah Murid SMP (Negeri+Swasta)']
            X_test = df_test[features]
            y_pred = model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            
            # --- Tampilan Gauge Chart & Penjelasan MAPE ---
            col1, col2 = st.columns([1, 1])
            with col1:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number", value = mape,
                    title = {'text': "Mean Absolute Percentage Error (MAPE)"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 20], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#003366"}, 'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 5], 'color': 'green'}, {'range': [5, 10], 'color': 'yellow'},
                            {'range': [10, 20], 'color': 'red'}],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 15}
                    }
                ))
                fig_gauge.update_layout(height=350)
                st.plotly_chart(fig_gauge, use_container_width=True)
            with col2:
                st.subheader("Apa Arti Akurasi Model Ini?")
                st.write(f"Nilai MAPE **{mape:.2f}%** berarti prediksi model memiliki rata-rata tingkat kesalahan sebesar {mape:.2f}%.")
                mape_rounded = int(round(mape, 0))
                st.write(f"""
                **Contoh dalam skenario nyata:**
                * Jika model memprediksi ada **100** murid di suatu kecamatan.
                * Kenyataannya, jumlah murid tersebut kemungkinan besar ada di antara **{100 - mape_rounded}** dan **{100 + mape_rounded}**.
                """)
                st.info("MAPE dihitung dengan melatih model pada data 2017-2023 dan mengujinya pada data 2024.")

            st.divider()
            
            # --- Plot Akurasi (Prediksi vs Aktual) ---
            st.subheader("Analisis Prediksi vs Aktual (Data Uji 2024)")
            plot_df = pd.DataFrame({'Aktual': y_test, 'Prediksi': y_pred, 'Kecamatan': df_test['Kecamatan']})
            fig_acc = px.scatter(
                plot_df, x='Aktual', y='Prediksi', hover_data=['Kecamatan'],
                title="Akurasi Model: Prediksi vs. Aktual (Tahun 2024)"
            )
            fig_acc.add_shape(type='line', x0=plot_df['Aktual'].min(), y0=plot_df['Aktual'].min(), 
                              x1=plot_df['Aktual'].max(), y1=plot_df['Aktual'].max(), 
                              line=dict(color='Red', dash='dash'))
            st.plotly_chart(fig_acc, use_container_width=True)
            
            # --- PERUBAHAN 1: Penjelasan Plot Akurasi ---
            st.markdown("""
            **Cara Membaca Grafik Ini:**
            * **Sumbu X (Aktual):** Jumlah murid sebenarnya di tahun 2024.
            * **Sumbu Y (Prediksi):** Jumlah murid yang *ditebak* oleh model.
            * **Garis Merah Putus-putus:** Garis ideal (Prediksi = Aktual).
            
            **Wawasan:** Semakin dekat titik-titik biru ke garis merah, semakin akurat prediksi model. 
            Titik yang jauh dari garis menunjukkan kecamatan di mana model mengalami kesulitan (mungkin karena anomali data di tahun tersebut).
            """)
            
        else:
            st.warning("Tidak ada data tahun 2024 atau data lag untuk melakukan pengujian akurasi.")

    else:
        # --- Tampilan untuk Kecamatan Spesifik ---
        st.header(f"Analisis Tren untuk: {selected_kecamatan}")
        
        data_hist = df[df['Kecamatan'] == selected_kecamatan].sort_values('Tahun')
        
        # Blok Metrik (Tidak berubah)
        try:
            data_2024 = data_hist[data_hist['Tahun'] == 2024]['Jumlah Murid SMP (Negeri+Swasta)'].iloc[0]
            data_2023 = data_hist[data_hist['Tahun'] == 2023]['Jumlah Murid SMP (Negeri+Swasta)'].iloc[0]
            pertumbuhan = ((data_2024 - data_2023) / data_2023) * 100
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Total Murid (2024)", value=f"{data_2024:,.0f}")
            col2.metric(label="Pertumbuhan vs 2023", value=f"{pertumbuhan:.2f} %", delta=f"{pertumbuhan:.2f} %")
            col3.metric(label="Total Murid (2023)", value=f"{data_2023:,.0f}")
        except Exception as e:
            st.info("Data historis tidak cukup untuk menampilkan perbandingan metrik.")
        
        # --- PERUBAHAN 2: Ringkasan Wawasan Dinamis (NLG) ---
        last_year_data = data_hist[data_hist['Tahun'] == 2024]
        
        if not last_year_data.empty:
            # --- Logika untuk menghasilkan prediksi (2025-2027) ---
            pred_list = []
            base_features_df = last_year_data[features].copy()
            current_lag_value = last_year_data['Jumlah Murid SMP (Negeri+Swasta)'].iloc[0]
            
            for t in range(2025, 2028): # Prediksi 3 tahun
                base_features_df['Tahun'] = t
                base_features_df['Jumlah Murid Tahun Lalu'] = current_lag_value
                pred = model.predict(base_features_df)[0]
                pred_list.append({'Tahun': t, 'Prediksi Murid': pred})
                current_lag_value = pred 
            
            df_pred = pd.DataFrame(pred_list)
            
            # --- Logika untuk tabel pertumbuhan (YoY) ---
            data_2024_row = pd.DataFrame([{'Tahun': 2024, 'Prediksi Murid': last_year_data['Jumlah Murid SMP (Negeri+Swasta)'].iloc[0]}])
            df_pred_calc = pd.concat([data_2024_row, df_pred], ignore_index=True).set_index('Tahun')
            df_pred_calc['% Pertumbuhan (YoY)'] = df_pred_calc['Prediksi Murid'].pct_change() * 100
            df_pred_tampil = df_pred_calc.drop(2024)
            
            # --- Mulai Tampilkan Wawasan Dinamis ---
            st.subheader("Ringkasan Wawasan (Insight)")
            
            val_2024 = data_2024_row.iloc[0]['Prediksi Murid']
            val_2027 = df_pred.iloc[-1]['Prediksi Murid'] # Ambil baris terakhir (2027)
            total_growth_percent = ((val_2027 - val_2024) / val_2024) * 100
            total_growth_abs = val_2027 - val_2024

            if total_growth_percent > 1:
                tren = "menunjukkan **tren pertumbuhan positif**"
            elif total_growth_percent < -1:
                tren = "menunjukkan **tren penurunan**"
            else:
                tren = "diprediksi **relatif stabil**"

            max_growth_year = df_pred_tampil['% Pertumbuhan (YoY)'].idxmax()
            max_growth_val = df_pred_tampil['% Pertumbuhan (YoY)'].max()

            st.markdown(f"""
            Berdasarkan model, Kecamatan **{selected_kecamatan}** {tren} selama 3 tahun ke depan.
            
            * Jumlah murid diprediksi akan berubah dari **{val_2024:,.0f}** di tahun 2024 menjadi **{val_2027:,.0f}** di tahun 2027.
            * Ini mewakili total perubahan sekitar **{total_growth_abs:,.0f} murid** (**{total_growth_percent:.2f}%**) selama 3 tahun.
            * Pertumbuhan tahunan tertinggi diprediksi terjadi pada tahun **{max_growth_year}**, dengan kenaikan sebesar **{max_growth_val:.2f}%** dari tahun sebelumnya.
            """)
            
            st.divider()

            # --- Bagian Plot dan Tabel (Tidak Berubah) ---
            st.subheader("1. Tren Historis (2016-2024)")
            fig_hist = px.line(data_hist, x='Tahun', y='Jumlah Murid SMP (Negeri+Swasta)', title=f"Tren Historis Murid SMP di {selected_kecamatan}", markers=True)
            fig_hist.update_xaxes(type='category')
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.subheader("Ringkasan Data Historis")
            tabel_hist = data_hist.set_index('Tahun')[['Jumlah Murid SMP (Negeri+Swasta)', 'Jumlah Guru SMP (Negeri+Swasta)', 'Jumlah Sekolah SMP (Negeri+Swasta)', 'Jumlah Murid Tahun Lalu']]
            st.dataframe(tabel_hist, use_container_width=True, column_config={
                "Jumlah Murid SMP (Negeri+Swasta)": st.column_config.BarChartColumn("Total Murid", width="medium"),
                "Jumlah Guru SMP (Negeri+Swasta)": st.column_config.NumberColumn("Total Guru", format="%.0f"),
                "Jumlah Sekolah SMP (Negeri+Swasta)": st.column_config.NumberColumn("Total Sekolah", format="%d"),
                "Jumlah Murid Tahun Lalu": st.column_config.NumberColumn("Murid Tahun Lalu", format="%.0f")
            })
            st.divider()

            st.subheader("2. Prediksi Tren Jangka Pendek (2025-2027)")
            df_pred_plot = df_pred.rename(columns={'Prediksi Murid': 'Jumlah Murid SMP (Negeri+Swasta)'})
            df_pred_plot['Tipe'] = 'Prediksi'
            data_hist_copy = data_hist[['Tahun', 'Jumlah Murid SMP (Negeri+Swasta)']].copy()
            data_hist_copy['Tipe'] = 'Historis'
            combined_df = pd.concat([data_hist_copy, df_pred_plot], ignore_index=True)
            
            fig_pred = px.line(combined_df, x='Tahun', y='Jumlah Murid SMP (Negeri+Swasta)', color='Tipe', markers=True, title=f"Prediksi Tren Murid SMP di {selected_kecamatan} hingga 2027")
            fig_pred.update_xaxes(type='category')
            st.plotly_chart(fig_pred, use_container_width=True)
            
            st.subheader("Tabel Prediksi dan Pertumbuhan (YoY)")
            min_growth = (df_pred_tampil['% Pertumbuhan (YoY)'].min() - 0.5) if not df_pred_tampil.empty else -1
            max_growth = (df_pred_tampil['% Pertumbuhan (YoY)'].max() + 0.5) if not df_pred_tampil.empty else 1
            
            st.dataframe(df_pred_tampil, use_container_width=True, column_config={
                "Prediksi Murid": st.column_config.NumberColumn("Prediksi Total Murid", format="%.0f"),
                "% Pertumbuhan (YoY)": st.column_config.ProgressColumn(
                    "Pertumbuhan (YoY)", format="%.2f%%", min_value=min_growth, max_value=max_growth
                )
            })

            with st.expander("Klik untuk melihat Metodologi Penelitian yang Digunakan"):
                st.markdown("""
                Penjelasan ini merinci metodologi yang digunakan untuk menghasilkan prediksi di atas, sesuai dengan draf skripsi Anda.
                
                1.  **Model Inti:** Prediksi dibuat menggunakan **Random Forest Regressor**, sebuah model *ensemble machine learning* yang menggabungkan banyak *decision tree* untuk menghasilkan prediksi yang lebih akurat dan stabil.
                
                2.  **Fitur Kunci (Rekayasa Fitur):** Fitur paling penting yang digunakan adalah **"Fitur Lag"** (`Jumlah Murid Tahun Lalu`). Ini berarti model dilatih untuk memprediksi jumlah murid tahun ini berdasarkan jumlah murid tahun lalu, serta faktor-faktor lain (jumlah guru dan sekolah).
                
                3.  **Proses Prediksi (Autoregresif):** Prediksi bersifat **rekursif** atau *walk-forward*.
                    * Prediksi 2025 dibuat menggunakan data aktual 2024.
                    * Prediksi 2026 dibuat menggunakan *hasil prediksi* 2025 sebagai input.
                    * Prediksi 2027 dibuat menggunakan *hasil prediksi* 2026 sebagai input.
                """)
        
        else:
            st.warning(f"Tidak ada data 2024 untuk {selected_kecamatan} yang dapat dijadikan basis prediksi.")