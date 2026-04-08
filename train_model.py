"""
train_model.py
==============
Pipeline Training Model SIBI AR Translator
-------------------------------------------
Skrip ini akan:
1. Baca dataset.json hasil dari collect_data.py
2. Build Dense Neural Network sederhana namun efektif
3. Train dengan augmentasi noise untuk robustness
4. Export model ke format TensorFlow.js (tfjs_model/)

Jalankan:
    python train_model.py

Requirements:
    pip install tensorflow tensorflowjs scikit-learn numpy
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import subprocess
import sys

# ── Konfigurasi ─────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
DATASET_FILE = os.path.join(BASE_DIR, "dataset.json")
MODEL_DIR    = os.path.join(BASE_DIR, "tfjs_model")
KERAS_PATH   = os.path.join(BASE_DIR, "sibi_model.keras")

EPOCHS       = 100
BATCH_SIZE   = 32
TEST_SIZE    = 0.15
NOISE_STD    = 0.02   # Data augmentation: tambah Gaussian noise
DROPOUT_RATE = 0.35

# ── Load Dataset ─────────────────────────────────────
def load_dataset():
    print("📂 Membaca dataset.json...")
    with open(DATASET_FILE, "r") as f:
        raw = json.load(f)
    
    data = raw["data"]
    metadata = raw.get("metadata", {})
    print(f"   Total sampel   : {len(data)}")
    print(f"   Jumlah kelas   : {metadata.get('num_classes', '?')}")
    print(f"   Label tersedia : {metadata.get('labels', [])}")
    
    X = np.array([d["landmarks"] for d in data], dtype=np.float32)
    y = np.array([d["label"] for d in data])
    
    return X, y

# ── Augmentasi Data ───────────────────────────────────
def augment_data(X, y, multiplier=3):
    """
    Tambahkan sampel augmentasi dengan Gaussian noise kecil.
    Ini membantu model menjadi lebih robust terhadap variasi pencahayaan
    dan pergerakan kamera.
    """
    X_aug_list = [X]
    y_aug_list = [y]
    
    for _ in range(multiplier):
        noise = np.random.normal(0, NOISE_STD, X.shape).astype(np.float32)
        X_noisy = np.clip(X + noise, -1.5, 1.5)
        X_aug_list.append(X_noisy)
        y_aug_list.append(y)
    
    return np.vstack(X_aug_list), np.concatenate(y_aug_list)

# ── Build Model ───────────────────────────────────────
def build_model(input_dim, num_classes):
    """
    Dense Neural Network:
    - Input: 63 fitur (21 landmarks * 3 koordinat xyz, ternormalisasi)
    - Hidden layers dengan BatchNorm dan Dropout untuk regularisasi
    - Output: softmax untuk multi-class classification
    """
    model = keras.Sequential([
        # Layer 1
        keras.layers.Dense(256, activation="relu", input_shape=(input_dim,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(DROPOUT_RATE),
        
        # Layer 2
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(DROPOUT_RATE),
        
        # Layer 3
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
        # Output
        keras.layers.Dense(num_classes, activation="softmax"),
    ], name="sibi_classifier")
    
    return model

# ── Main ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  SIBI AR Translator — Model Training Script")
    print("=" * 55 + "\n")
    
    if not os.path.exists(DATASET_FILE):
        print(f"❌ ERROR: dataset.json tidak ditemukan!")
        print(f"   Jalankan 'python collect_data.py' terlebih dahulu.")
        return
    
    # 1. Load data
    X, y_raw = load_dataset()
    
    # 2. Encode labels ke integer
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)
    labels = encoder.classes_.tolist()
    
    print(f"\n🏷️  Label urutan model: {labels}")
    
    # 3. Augmentasi
    print(f"\n🔄 Augmentasi data (3x noise Gaussian std={NOISE_STD})...")
    X_aug, y_aug = augment_data(X, y)
    print(f"   Sampel setelah augmentasi: {len(X_aug)}")
    
    # 4. Train/Val split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X_aug, y_aug, test_size=TEST_SIZE, random_state=42, stratify=y_aug
    )
    print(f"   Train: {len(X_train)} | Val: {len(X_val)}")
    
    # 5. Build model
    print(f"\n🧠 Membangun model...")
    model = build_model(input_dim=63, num_classes=num_classes)
    model.summary()
    
    # 6. Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # 7. Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            KERAS_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0
        ),
    ]
    
    # 8. Train
    print(f"\n🚀 Mulai training ({EPOCHS} epochs max)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 9. Evaluasi
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n{'='*55}")
    print(f"✅ Akurasi Validasi Final : {val_acc*100:.2f}%")
    print(f"   Loss Validasi Final   : {val_loss:.4f}")
    
    # 10. Simpan label order ke JSON (penting untuk inference!)
    labels_path = os.path.join(BASE_DIR, "labels.json")
    with open(labels_path, "w") as f:
        json.dump({"labels": labels}, f)
    print(f"💾 Label order disimpan ke: {labels_path}")
    
    # 11. Export ke TensorFlow.js format
    print(f"\n📦 Mengonversi ke format TensorFlow.js...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(model, MODEL_DIR)
        print(f"✅ Model TF.js disimpan ke: {MODEL_DIR}/")
    except ImportError:
        # Fallback: gunakan CLI tensorflowjs_converter
        print("   tensorflowjs Python package tidak ada, mencoba CLI...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "tensorflowjs.converters.converter",
                "--input_format=keras",
                KERAS_PATH,
                MODEL_DIR
            ], capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"✅ Model TF.js disimpan ke: {MODEL_DIR}/")
            else:
                # Fallback 2: tensorflowjs_converter command
                result2 = subprocess.run([
                    "tensorflowjs_converter",
                    "--input_format=keras",
                    KERAS_PATH,
                    MODEL_DIR
                ], capture_output=True, text=True, timeout=120)
                if result2.returncode == 0:
                    print(f"✅ Model TF.js disimpan ke: {MODEL_DIR}/")
                else:
                    print(f"⚠️  Konversi otomatis gagal. Jalankan manual:")
                    print(f"   tensorflowjs_converter --input_format=keras {KERAS_PATH} {MODEL_DIR}")
        except Exception as e:
            print(f"⚠️  Konversi gagal: {e}")
            print(f"   Jalankan manual: tensorflowjs_converter --input_format=keras {KERAS_PATH} {MODEL_DIR}")
    
    print(f"\n✅ SELESAI!")
    print(f"   Buka sibi_ar.html di browser untuk mencoba model!")

if __name__ == "__main__":
    main()
