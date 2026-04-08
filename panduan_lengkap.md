# 📖 Panduan Lengkap SIBI AR Translator

Sistem penerjemah Bahasa Isyarat SIBI (Sistem Isyarat Bahasa Indonesia) berbasis AI yang berjalan di browser menggunakan **MediaPipe Hands** + **TensorFlow.js**.

---

## 🗂️ Struktur Proyek

```
SIBI AR Translator/
├── SIBI/               ← Dataset gambar (A-Y, 24 kelas)
├── collect_data.py     ← Step 1: Ekstrak landmark dari gambar
├── train_model.py      ← Step 2: Training model neural network
├── sibi_ar.html        ← Step 3: Aplikasi AR utama
├── dataset.json        ← (otomatis) hasil dari collect_data.py
├── labels.json         ← (otomatis) urutan label kelas
├── sibi_model.keras    ← (otomatis) model tersimpan
└── tfjs_model/         ← (otomatis) model TF.js untuk browser
    ├── model.json
    └── group1-shard*.bin
```

---

## ⚙️ Persyaratan

### Python (versi 3.8–3.11 disarankan)

Install paket yang diperlukan:
```bash
pip install mediapipe opencv-python pillow tqdm
pip install tensorflow==2.15.0
pip install tensorflowjs
pip install scikit-learn numpy
```

> **Catatan**: Jika menggunakan GPU, install `tensorflow-gpu` sebagai pengganti `tensorflow`.

---

## 🚀 Langkah-Langkah Pipeline

### Step 1 — Pengumpulan Data (`collect_data.py`)

Skrip ini akan memproses semua gambar di folder `SIBI/`, mendeteksi landmark tangan menggunakan MediaPipe, dan menyimpannya ke `dataset.json`.

```bash
cd "c:\Rafen\Kelas 12 SMK\Kecerdasan Artifisial 2\SIBI AR Translator"
python collect_data.py
```

**Output yang diharapkan:**
```
✅ Ditemukan 24 kelas: ['A', 'B', 'C', ...]
📂 Kelas 'A': 220 gambar
  A: 100%|████████████████| 220/220
...
✅ Terdeteksi : 4200 sampel
⚠️  Dilewati   : 800 gambar (landmark tidak terdeteksi)
💾 Dataset disimpan ke: dataset.json
```

> **Normal jika banyak gambar dilewati** — beberapa gambar mungkin blur, sudut terlalu ekstrem, atau tangan tidak terlihat jelas oleh MediaPipe.

---

### Step 2 — Training Model (`train_model.py`)

Skrip ini akan membaca `dataset.json`, melatih neural network, dan mengekspor model ke format TF.js.

```bash
python train_model.py
```

**Output yang diharapkan:**
```
📂 Membaca dataset.json...
   Total sampel   : 4200
   Jumlah kelas   : 24
🔄 Augmentasi data (3x noise Gaussian)...
   Sampel setelah augmentasi: 16800
🧠 Membangun model...
🚀 Mulai training (100 epochs max)...
Epoch 1/100: loss=2.8 accuracy=0.25
...
Epoch 47/100: loss=0.08 accuracy=0.98 ← EarlyStopping akan berhenti di sini
✅ Akurasi Validasi Final : 96.50%
💾 label order disimpan ke: labels.json
📦 Mengonversi ke format TensorFlow.js...
✅ Model TF.js disimpan ke: tfjs_model/
```

> **Target akurasi yang baik**: ≥ 90% validasi. Jika di bawah itu, coba tambah epoch atau kurangi `DROPOUT_RATE` di `train_model.py`.

---

### Step 3 — Jalankan Aplikasi AR (`sibi_ar.html`)

> ⚠️ **PENTING**: File harus dibuka melalui **HTTP server**, bukan langsung di browser (`file://`), karena TF.js memerlukan CORS headers untuk memuat `model.json`.

#### Cara termudah — Python HTTP Server:
```bash
cd "c:\Rafen\Kelas 12 SMK\Kecerdasan Artifisial 2\SIBI AR Translator"
python -m http.server 8080
```

Kemudian buka di browser:
```
http://localhost:8080/sibi_ar.html
```

#### Alternatif — VS Code Live Server:
Install ekstensi **Live Server** di VS Code, klik kanan `sibi_ar.html` → **Open with Live Server**.

---

## 🎮 Cara Penggunaan Aplikasi

### Antarmuka
| Area | Fungsi |
|------|--------|
| **Kamera (kiri)** | Feed webcam dengan AR overlay tangan |
| **Huruf AR besar** | Prediksi huruf SIBI yang terdeteksi (melayang di atas tangan) |
| **Sidebar (kanan)** | Hasil terjemahan, progress bar, tombol aksi |
| **Grid alfabet** | Menampilkan huruf yang aktif terdeteksi |

### Cara Mengetik
1. **Tunjukkan gestur tangan** ke kamera (24 huruf: A–Y kecuali J & Z)
2. **Tahan posisi selama 1.5 detik** → huruf otomatis terketik
3. **Progress bar** menunjukkan seberapa lama kamu sudah menahan

### Tombol Manual
| Tombol | Fungsi |
|--------|--------|
| `⎵ Spasi` | Tambahkan spasi antar kata |
| `✕ Hapus` | Hapus seluruh teks |

### Status Indikator (pojok kanan atas)
| Indikator | Warna | Arti |
|-----------|-------|------|
| KAMERA | 🟢 Hijau | Webcam aktif |
| MODEL | 🟢 Hijau | AI model berhasil dimuat |
| MODEL | 🔴 Merah | Model belum dilatih / tidak ditemukan |
| TANGAN | 🟢 Hijau | Tangan terdeteksi |
| MEMBACA | 🟡 Kuning | Sedang menghitung hold timer |

---

## 🔧 Troubleshooting

### ❗ `labels.json` atau `tfjs_model/model.json` tidak ditemukan
→ Jalankan `collect_data.py` dan `train_model.py` terlebih dahulu.

### ❗ Banyak gambar dilewati saat `collect_data.py`
→ Normal sampai 30–40% gambar dilewati. Jika > 70%, coba turunkan `min_detection_confidence` dari `0.3` ke `0.1` di `collect_data.py`.

### ❗ Akurasi model rendah (< 80%)
→ Beberapa huruf SIBI memiliki pose tangan yang mirip. Coba:
- Tambah `EPOCHS = 150` di `train_model.py`  
- Kurangi `DROPOUT_RATE = 0.25`
- Tambah data augmentasi (`multiplier=5`)

### ❗ Browser memblokir kamera
→ Pastikan akses via `http://localhost:8080` bukan `file://...`

### ❗ Confidence selalu rendah saat pakai webcam
→ Pastikan:
- Ruangan cukup terang
- Tangan kontras dengan latar belakang
- Jarak tangan 40–80 cm dari kamera

---

## 📊 Arsitektur Model

```
Input: 63 fitur (21 landmark × 3 koordinat xyz, dinormalisasi)
    ↓
Dense(256, relu) → BatchNorm → Dropout(0.35)
    ↓
Dense(128, relu) → BatchNorm → Dropout(0.35)
    ↓
Dense(64, relu)  → BatchNorm → Dropout(0.20)
    ↓
Dense(24, softmax)  ← Output: probabilitas 24 kelas
```

**Normalisasi landmark:**
1. Kurangi semua titik dengan koordinat wrist (titik 0) → posisi relatif
2. Bagi dengan jarak Euclidean terjauh → scale invariant
3. Hasil: 63 nilai float dalam range [-1, 1]

---

## 💡 Catatan Huruf J & Z

Huruf **J** dan **Z** dalam SIBI melibatkan **gerakan** (bukan pose statis), sehingga tidak bisa direpresentasikan dengan satu gambar diam. Dataset yang ada tidak memiliki folder J dan Z.

**Opsi ke depan:**
- Implementasikan deteksi gerakan (trajectory tracking) untuk J dan Z
- Gunakan input manual via keyboard untuk J dan Z

---

*Dibuat untuk proyek Kecerdasan Artifisial 2 · SMK · 2024*
