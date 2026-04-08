"""
export_tfjs.py
==============
Manual exporter: sibi_model.keras -> tfjs_model/
Bekerja di Python 3.13 + TF 2.20 tanpa tensorflowjs package.

Cara kerja:
- Load model Keras
- Ekstrak arsitektur dan bobot
- Simpan ke format TF.js (model.json + binary weight shard)

Jalankan:
    python export_tfjs.py
"""

import os, json, struct, numpy as np
import tensorflow as tf

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
KERAS_PATH  = os.path.join(BASE_DIR, "sibi_model.keras")
OUTPUT_DIR  = os.path.join(BASE_DIR, "tfjs_model")
SHARD_SIZE  = 4 * 1024 * 1024  # 4 MB per shard

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("  SIBI Model Exporter — Keras -> TF.js format")
print("=" * 55 + "\n")

# Load model
print("[LOAD] Loading sibi_model.keras...")
model = tf.keras.models.load_model(KERAS_PATH)
model.summary()

# ── Kumpulkan semua bobot ──────────────────────────────
weight_data = []
weight_manifests_entries = []

all_weights_bytes = b""

for layer in model.layers:
    layer_weights = layer.get_weights()
    if not layer_weights:
        continue
    
    for i, w in enumerate(layer_weights):
        arr = w.astype(np.float32)
        
        # Penamaan bobot sesuai tipe layer untuk kecocokan TF.js
        if "BatchNormalization" in layer.__class__.__name__:
            names = ["gamma", "beta", "moving_mean", "moving_variance"]
            weight_name = f"{layer.name}/{names[i]}" if i < 4 else f"{layer.name}/weight_{i}"
        else:
            names = ["kernel", "bias"]
            weight_name = f"{layer.name}/{names[i]}" if i < 2 else f"{layer.name}/weight_{i}"
        
        byte_data = arr.tobytes()
        byte_length = len(byte_data)
        
        weight_manifests_entries.append({
            "name": weight_name,
            "shape": list(arr.shape),
            "dtype": "float32",
            "byteLength": byte_length
        })
        
        all_weights_bytes += byte_data
        print(f"   [W] {weight_name}: shape={arr.shape}, bytes={byte_length}")

# ── Simpan binary shard ────────────────────────────────
shard_paths = []
shard_sizes = []
num_shards  = max(1, (len(all_weights_bytes) + SHARD_SIZE - 1) // SHARD_SIZE)

for i in range(num_shards):
    start = i * SHARD_SIZE
    end   = min(start + SHARD_SIZE, len(all_weights_bytes))
    chunk = all_weights_bytes[start:end]
    
    shard_filename = f"group1-shard{i+1}of{num_shards}.bin"
    shard_path     = os.path.join(OUTPUT_DIR, shard_filename)
    
    with open(shard_path, "wb") as f:
        f.write(chunk)
    
    shard_paths.append(shard_filename)
    shard_sizes.append(len(chunk))
    print(f"   [BIN] Saved: {shard_filename} ({len(chunk)} bytes)")

# ── Build manifests ────────────────────────────────────
# Semua bobot dalam satu manifest entry yang merujuk semua shard
manifest = [{
    "paths": shard_paths,
    "weights": weight_manifests_entries
}]

# ── Bangun model.json ──────────────────────────────────
model_config = model.get_config()

# Inject batch_input_shape into the first layer config to prevent TF.js format error
if "layers" in model_config and len(model_config["layers"]) > 0:
    first_layer = model_config["layers"][0]
    if "config" in first_layer:
        # Hapus semua kemungkinan argumen input shape ganda yang dihasilkan Keras 3
        # karena TF.js akan melemparkan error "not both at the same time"
        for key in ["batch_shape", "batch_input_shape", "batchInputShape", "input_shape", "inputShape"]:
            first_layer["config"].pop(key, None)
        
        # Tambahkan secara eksplisit 1 pasang argumen yang diterima TF.js
        first_layer["config"]["batchInputShape"] = [None, 63]

# Keras 3 to Keras 2 compatibility cleaner for TFJS
def clean_keras3_config(obj):
    if isinstance(obj, dict):
        # Keras 3 saves dtype as a DTypePolicy config (e.g. {"class_name": "DTypePolicy", "config": {"name": "float32"}})
        # TFJS expects dtype to be just a string (e.g. "float32")
        if obj.get("class_name") == "DTypePolicy" and "config" in obj:
            return obj["config"].get("name", "float32")
        
        for k, v in list(obj.items()):
            if k == 'dtype' and isinstance(v, dict):
                if 'config' in v and 'name' in v['config']:
                    obj[k] = v['config']['name']
                else:
                    obj[k] = 'float32'
            else:
                obj[k] = clean_keras3_config(v)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = clean_keras3_config(obj[i])
    return obj

model_config = clean_keras3_config(model_config)

model_json = {
    "format": "layers-model",
    "generatedBy": f"keras v{tf.keras.__version__}",
    "convertedBy": "Manual SIBI Exporter v1.0",
    "modelTopology": {
        "class_name": model.__class__.__name__,
        "config": model_config,
        "keras_version": tf.keras.__version__,
        "backend": "tensorflow"
    },
    "weightsManifest": manifest
}

model_json_path = os.path.join(OUTPUT_DIR, "model.json")
with open(model_json_path, "w") as f:
    json.dump(model_json, f)

model_json_size = os.path.getsize(model_json_path)
print(f"\n[SAVE] model.json saved ({model_json_size/1024:.1f} KB)")

# ── Simpan labels ──────────────────────────────────────
labels_src  = os.path.join(BASE_DIR, "labels.json")
if os.path.exists(labels_src):
    import shutil
    shutil.copy(labels_src, os.path.join(OUTPUT_DIR, "labels.json"))
    print("[SAVE] labels.json copied to tfjs_model/")

# ── Verifikasi ────────────────────────────────────────
files = os.listdir(OUTPUT_DIR)
total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in files)
print(f"\n[OK] tfjs_model/ contents ({total_size/1024:.1f} KB total):")
for fname in sorted(files):
    fsize = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
    print(f"     {fname}: {fsize/1024:.1f} KB")

print("\n[DONE] Export selesai! Buka sibi_ar.html di browser.")
