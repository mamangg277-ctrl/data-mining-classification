# Data Classifier

A browser-based, desktop, and command-line tool for **binary and multiclass classification** of tabular data. Load a CSV, Excel, or JSON file, pick a target column with two or more classes (for example `true`/`false`, `A`/`B`, or `low`/`medium`/`high`), and train, compare, evaluate, and use classifiers without writing code. **Works fully offline** - all processing and history stay on your machine.

---

## 🚀 Panduan Cepat / Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Kaa278/data-mining-classification.git
cd data-mining-classification
```

### 2. Setup Virtual Environment

* **Windows (PowerShell / Command Prompt):**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  pip install -e .
  ```

* **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e .
  ```

*(Catatan: Pengguna Linux yang ingin menjalankan Desktop GUI Tkinter membutuhkan package sistem: `sudo apt install python3-tk`)*

---

### 3. Cara Menjalankan

#### A. Web App (Sangat Direkomendasikan 🌐)
Buka terminal di direktori proyek (pastikan virtual environment sudah aktif), lalu jalankan:

```bash
dc web
```
*atau jika shortcut `dc` belum terdaftar di PATH:*
```bash
python -m dc.web.app
```
Aplikasi web otomatis terbuka di browser pada alamat **`http://127.0.0.1:5000`**.

#### B. Desktop GUI (Tkinter 🖥️)
```bash
dc gui
```
*atau:*
```bash
python -m dc.gui.app
```

#### C. Command Line Interface (CLI 💻)
```bash
dc --help
```

---

## 📖 Alur Penggunaan Aplikasi (Workflow)

1. **Tab Data:**
   * Upload file dataset tabular kamu (format `.csv`, `.xlsx`, atau `.json`). Contoh dataset bawaan tersedia di `data.csv`.
   * Pilih kolom target klasifikasi: 2 kelas (biner) atau lebih (multikelas).
2. **Tab Explore (EDA):**
   * Periksa statistik deskriptif, missing values, dan distribusi kelas target. Laporan tersimpan otomatis di `reports/`.
3. **Tab Train:**
   * Atur rasio test size, k-fold cross validation, dan pilih algoritma (*Logistic Regression*, *Random Forest*, *Gradient Boosting*).
   * Klik **Train Model** (proses training berjalan di background).
4. **Tab Evaluate:**
   * Lihat metrik performa (*Accuracy*, *Precision*, *Recall*, *F1-Score*, *ROC-AUC*, *Confusion Matrix*).
   * Untuk model biner, geser slider probabilitas threshold untuk melihat pengaruhnya terhadap klasifikasi.
   * Untuk model multikelas, lihat metrik macro/weighted, metrik per kelas, dan confusion matrix N×N.
5. **Tab Predict:**
   * Lakukan prediksi tunggal (masukkan nilai fitur secara manual).
   * Model biner menampilkan probabilitas kelas positif dan threshold; model multikelas menampilkan kelas terprediksi beserta probabilitas setiap kelas.
   * Atau lakukan prediksi massal (*batch prediction*) dengan mengunggah file data baru dan unduh hasilnya ke CSV/Excel.
6. **Tab History & Storage:**
   * Pantau riwayat analisis dan kelola ruang penyimpanan lokal.

########################
Link Kaggle yang dapat digunakan untuk menguji pengklasifikasian
https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels

Link Folder Google Drive
https://drive.google.com/drive/folders/17B97DEmeMwzQtWqhkix3UnUhbwKMAmus?usp=sharing 

---

## 🛠️ Tech Stack

* **Language:** Python >= 3.10
* **Machine Learning:** `scikit-learn`, `pandas`, `numpy`, `joblib`
* **Web App:** Flask, Bootstrap CSS, HTMX, SQLite
* **Desktop App:** Tkinter, Matplotlib
* **CLI:** Typer, Rich

---

## 🧪 Testing & Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```
