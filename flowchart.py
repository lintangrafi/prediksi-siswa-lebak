import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(12, 14))
    
    # Definisi Kotak (Box)
    # Format: (x, y, width, height, label, color)
    boxes = {
        "start": (0.4, 0.9, 0.2, 0.05, "Mulai: Data BPS (JSON)", "#d9d9d9"),
        "preprocess": (0.3, 0.8, 0.4, 0.05, "Preprocessing:\nCleaning & Feature Engineering\n(Buat Fitur 'Lag' Tahun Lalu)", "#add8e6"),
        "split": (0.3, 0.7, 0.4, 0.05, "Split Data Strategy\nTrain: < 2024 | Test: = 2024", "#ffcccb"),
        "pipeline_rf": (0.1, 0.55, 0.35, 0.08, "Pipeline Random Forest:\n1. Imputer (Median)\n2. Scaler & OneHot\n3. RF Regressor (n_est=200)", "#90ee90"),
        "pipeline_lr": (0.55, 0.55, 0.35, 0.08, "Pipeline Linear Regression:\n1. Imputer (Median)\n2. Scaler & OneHot\n3. Linear Regression", "#ffd700"),
        "predict": (0.3, 0.4, 0.4, 0.05, "Prediksi pada Data Test (2024)", "#e0ffff"),
        "eval": (0.3, 0.3, 0.4, 0.05, "Evaluasi Komparasi:\nHitung MAPE & Akurasi", "#dda0dd"),
        "viz": (0.3, 0.2, 0.4, 0.05, "Visualisasi Streamlit:\nGrafik Tren & Tabel Prediksi", "#87cefa"),
        "end": (0.4, 0.1, 0.2, 0.05, "Selesai / Output", "#d9d9d9")
    }

    # Gambar Kotak
    for key, (x, y, w, h, label, color) in boxes.items():
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec="black", fc=color)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=10, weight="bold")

    # Gambar Panah (Alur)
    arrows = [
        ("start", "preprocess"),
        ("preprocess", "split"),
        ("split", "pipeline_rf"), # Cabang Kiri
        ("split", "pipeline_lr"), # Cabang Kanan
        ("pipeline_rf", "predict"),
        ("pipeline_lr", "predict"),
        ("predict", "eval"),
        ("eval", "viz"),
        ("viz", "end")
    ]

    for start, end in arrows:
        # Koordinat Awal (Bawah kotak start)
        sx, sy, sw, sh, _, _ = boxes[start]
        start_x = sx + sw/2
        start_y = sy

        # Koordinat Akhir (Atas kotak end)
        ex, ey, ew, eh, _, _ = boxes[end]
        end_x = ex + ew/2
        end_y = ey + eh

        # Logika panah bercabang
        if start == "split":
            if end == "pipeline_rf": start_x -= 0.1 # Geser kiri dikit
            if end == "pipeline_lr": start_x += 0.1 # Geser kanan dikit
        
        ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle="->", lw=2))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.title("Arsitektur Sistem Prediksi & Komparasi Model (BAB III)", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig("diagram_alur_penelitian.png")
    plt.show()

draw_flowchart()