# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
collect_data.py
===============
Pipeline Pengumpulan Data SIBI AR Translator
--------------------------------------------
Skrip ini akan:
1. Download hand_landmarker.task (model MediaPipe baru)
2. Loop setiap folder alfabet di ./SIBI/
3. Gunakan MediaPipe HandLandmarker (API baru v0.10+) untuk mendeteksi landmark
4. Normalisasi koordinat (kurangi wrist, bagi jarak max)
5. Simpan hasilnya ke dataset.json

Jalankan:
    python collect_data.py

Requirements:
    pip install mediapipe opencv-python pillow tqdm
"""

import os
import json
import math
import urllib.request
import cv2
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Konfigurasi ─────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SIBI_DIR    = os.path.join(BASE_DIR, "SIBI")
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset.json")
MODEL_FILE  = os.path.join(BASE_DIR, "hand_landmarker.task")
MODEL_URL   = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Download model jika belum ada ───────────────────
def ensure_model():
    if os.path.exists(MODEL_FILE):
        size_kb = os.path.getsize(MODEL_FILE) / 1024
        print(f"[OK] Model sudah ada: hand_landmarker.task ({size_kb:.0f} KB)")
        return
    print("[DL] Mengunduh hand_landmarker.task (~29 MB)...")
    print(f"     URL: {MODEL_URL}")
    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                pct = int(count * block_size * 100 / total_size)
                print(f"     Progress: {min(pct,100)}%", end='\r')
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE, reporthook)
        print()
        size_kb = os.path.getsize(MODEL_FILE) / 1024
        print(f"[OK] Download selesai ({size_kb:.0f} KB)")
    except Exception as e:
        print(f"[ERROR] Download gagal: {e}")
        print("   Coba download manual dari:")
        print(f"   {MODEL_URL}")
        print(f"   Simpan ke: {MODEL_FILE}")
        raise

# ── Build HandLandmarker ─────────────────────────────
def create_detector():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_FILE)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(options)

# ── Normalisasi Landmark ─────────────────────────────
def normalize_landmarks(hand_landmarks):
    """
    Normalisasi 21 titik landmark tangan:
    1. Jadikan titik 0 (wrist) sebagai origin
    2. Bagi dengan jarak Euclidean terjauh → scale-invariant
    Output: list flat 63 nilai [x0,y0,z0, ..., x20,y20,z20]
    hand_landmarks: list of NormalizedLandmark (mediapipe Tasks API v0.10+)
    """
    lm = hand_landmarks  # Already a list of NormalizedLandmark

    wrist_x = lm[0].x
    wrist_y = lm[0].y
    wrist_z = lm[0].z

    # Geser ke origin (wrist = 0,0,0)
    shifted = [(l.x - wrist_x, l.y - wrist_y, l.z - wrist_z) for l in lm]

    # Cari jarak max
    max_dist = max(math.sqrt(x**2 + y**2 + z**2) for x, y, z in shifted)
    if max_dist < 1e-6:
        max_dist = 1e-6

    # Flatten
    result = []
    for (x, y, z) in shifted:
        result.extend([x / max_dist, y / max_dist, z / max_dist])

    return result  # 63 nilai float

# ── Proses satu gambar ───────────────────────────────
def process_image(detector, img_path):
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        return None

    # Resize jika terlalu besar
    h, w = img_cv.shape[:2]
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)))

    # Convert BGR → RGB untuk MediaPipe
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    return normalize_landmarks(result.hand_landmarks[0])

# ── Loop Dataset ─────────────────────────────────────
def collect_all(detector):
    dataset  = []
    skipped  = 0
    detected = 0

    labels = sorted([
        d for d in os.listdir(SIBI_DIR)
        if os.path.isdir(os.path.join(SIBI_DIR, d))
    ])
    print(f"[OK] Ditemukan {len(labels)} kelas: {labels}\n")

    for label in labels:
        folder = os.path.join(SIBI_DIR, label)
        images = [
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ]
        print(f"[DIR] Kelas '{label}': {len(images)} gambar")

        label_count = 0
        for img_file in tqdm(images, desc=f"  {label}", ncols=72, leave=False):
            img_path = os.path.join(folder, img_file)
            normalized = process_image(detector, img_path)

            if normalized is None:
                skipped += 1
                continue

            dataset.append({"label": label, "landmarks": normalized})
            detected += 1
            label_count += 1

        print(f"   -> {label}: {label_count} sampel terdeteksi")

    return dataset, detected, skipped, labels

# ── Main ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  SIBI AR Translator -- Data Collection Script")
    print("  MediaPipe HandLandmarker API (v0.10+)")
    print("=" * 60 + "\n")

    if not os.path.isdir(SIBI_DIR):
        print(f"[ERROR] Folder SIBI tidak ditemukan di:\n   {SIBI_DIR}")
        return

    # Download model jika belum ada
    ensure_model()
    print()

    # Buat detector
    print("[...] Membuat HandLandmarker detector...")
    detector = create_detector()
    print("[OK] Detector siap!\n")

    # Proses semua gambar
    dataset, detected, skipped, labels = collect_all(detector)
    detector.close()

    total = detected + skipped
    pct_det = detected / total * 100 if total > 0 else 0

    # Statistik
    print(f"\n{'='*60}")
    print(f"[OK]   Terdeteksi : {detected} sampel")
    print(f"[SKIP] Dilewati   : {skipped} gambar (landmark tidak terdeteksi)")
    print(f"[INFO] Total      : {total} gambar diproses")
    print(f"       Tingkat deteksi: {pct_det:.1f}%")

    if not dataset:
        print("\n[ERROR] Dataset kosong!")
        return

    # Distribusi per kelas
    from collections import Counter
    dist = Counter(d["label"] for d in dataset)
    print(f"\n[INFO] Distribusi sampel per kelas:")
    for lbl in sorted(dist.keys()):
        bar = "#" * min(30, dist[lbl] // 10)
        print(f"   {lbl:>2}: {dist[lbl]:>4} sampel  {bar}")

    # Simpan JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "metadata": {
                "total_samples": detected,
                "num_classes": len(dist),
                "labels": sorted(dist.keys()),
                "feature_size": 63,
                "description": "SIBI hand landmarks: 21 pts * 3 coords (x,y,z), wrist-normalized"
            },
            "data": dataset
        }, f)

    size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"\n[SAVE] Dataset disimpan ke: {OUTPUT_FILE}")
    print(f"       Ukuran file: {size_kb:.1f} KB ({size_kb/1024:.2f} MB)")
    print(f"\n[DONE] SELESAI! Lanjutkan dengan: python train_model.py")

if __name__ == "__main__":
    main()
