"""
main.py - Plant Disease Detection API
Auto-download model từ Google Drive nếu chưa có
"""

import json, sys, io, zipfile, tempfile, shutil, requests
from pathlib import Path

import numpy as np
from PIL import Image

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
keras = tf.keras
print(f"[INFO] TensorFlow {tf.__version__} | Keras {keras.__version__}")

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_PATH   = Path("model/plant_disease_model2.keras")
CLASSES_PATH = Path("model/classes.txt")
IMG_SIZE     = (224, 224)

# Google Drive file ID
GDRIVE_FILE_ID = "1Hbrwc3u0GPr-0MahbG_YMOGRZOeKoFKh"

# ─────────────────────────────────────────────
# Download model từ Google Drive
# ─────────────────────────────────────────────
def download_from_gdrive(file_id: str, dest: Path):
    print(f"[INFO] Đang download model từ Google Drive...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    # Dùng export link mới của Google Drive (hỗ trợ file lớn)
    URL = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

    response = session.get(URL, stream=True)

    total = 0
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                total += len(chunk)
                print(f"[INFO] Downloaded {total // (1024*1024)} MB...", end="\r")

    print(f"\n[OK] Download xong: {dest} ({total // (1024*1024)} MB)")


# ─────────────────────────────────────────────
# Patch config
# ─────────────────────────────────────────────
def clean_node(node):
    if not isinstance(node, dict):
        return node
    node.pop("quantization_config", None)
    if node.get("class_name") == "DTypePolicy":
        node.get("config", {}).pop("shared_object_id", None)
    node.pop("shared_object_id", None)
    for k, v in node.items():
        if isinstance(v, dict):
            node[k] = clean_node(v)
        elif isinstance(v, list):
            node[k] = [clean_node(i) if isinstance(i, dict) else i for i in v]
    return node


def patch_config(config: dict) -> dict:
    config = clean_node(config)
    layers = config.get("config", {}).get("layers", [])
    if not layers:
        return config

    new_layers = []
    td_input_tensor = None

    for layer in layers:
        cn = layer.get("class_name", "")
        if cn == "TrueDivide":
            args = layer.get("inbound_nodes", [{}])[0].get("args", [])
            if args and isinstance(args[0], dict):
                td_input_tensor = args[0]
            continue
        if cn == "Subtract":
            rescaling = {
                "module": "keras.layers",
                "class_name": "Rescaling",
                "config": {
                    "name": "rescaling_preprocess",
                    "trainable": False,
                    "dtype": "float32",
                    "scale": float(1.0 / 127.5),
                    "offset": float(-1.0),
                },
                "registered_name": None,
                "build_config": {"input_shape": [None, 224, 224, 3]},
                "name": "rescaling_preprocess",
                "inbound_nodes": [
                    {"args": [td_input_tensor], "kwargs": {}}
                ] if td_input_tensor else [],
            }
            new_layers.append(rescaling)
            continue
        for node in layer.get("inbound_nodes", []):
            for arg in node.get("args", []):
                if isinstance(arg, dict):
                    hist = arg.get("config", {}).get("keras_history", [])
                    if hist and hist[0] == "subtract":
                        hist[0] = "rescaling_preprocess"
        new_layers.append(layer)

    config["config"]["layers"] = new_layers
    return config


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
def load_model_patched(path: Path):
    tmpdir = Path(tempfile.mkdtemp())
    try:
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(tmpdir)
        cfg_path = tmpdir / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        print("[INFO] Patching config...")
        cfg = patch_config(cfg)
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        tmp_keras = tmpdir / "patched.keras"
        with zipfile.ZipFile(tmp_keras, "w", zipfile.ZIP_STORED) as zf:
            for f in tmpdir.iterdir():
                if f.name != "patched.keras":
                    zf.write(f, f.name)
        model = keras.models.load_model(str(tmp_keras), compile=False)
        print("[OK] Model loaded!")
        return model
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def load_model_safe(path: Path):
    try:
        return load_model_patched(path)
    except Exception as e:
        print(f"[WARN] Patch failed: {e}")

    try:
        print("[INFO] Strategy 2: rebuild architecture...")
        inp  = keras.Input(shape=(224, 224, 3), name="image_rgb_0_255")
        x    = keras.layers.Rescaling(scale=1./127.5, offset=-1.0)(inp)
        base = keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights=None, pooling="avg"
        )
        x   = base(x)
        x   = keras.layers.Dropout(0.2)(x)
        x   = keras.layers.Dense(256, activation="relu", name="dense")(x)
        x   = keras.layers.Dropout(0.2)(x)
        out = keras.layers.Dense(38, activation="softmax", name="class_probs")(x)
        m   = keras.Model(inp, out)
        tmpdir = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(str(path), "r") as zf:
            zf.extract("model.weights.h5", tmpdir)
        m.load_weights(str(tmpdir / "model.weights.h5"), by_name=True, skip_mismatch=True)
        shutil.rmtree(tmpdir, ignore_errors=True)
        print("[OK] Strategy 2 OK!")
        return m
    except Exception as e:
        raise RuntimeError(f"Không load được model: {e}")


# ─────────────────────────────────────────────
# Load classes
# ─────────────────────────────────────────────
def load_classes() -> list:
    if CLASSES_PATH.exists():
        lines = CLASSES_PATH.read_text(encoding="utf-8").splitlines()
        result = [l.strip() for l in lines if l.strip()]
        if result:
            return result
    json_path = Path("model/class_indices.json")
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return [k for k, _ in sorted(data.items(), key=lambda x: x[1])]
    print("[WARN] Không tìm thấy class names → dùng fallback")
    return [f"Class_{i}" for i in range(38)]


# ─────────────────────────────────────────────
# Khởi động — tự download model nếu chưa có
# ─────────────────────────────────────────────
if not MODEL_PATH.exists():
    print(f"[INFO] Model chưa có, đang download...")
    try:
        download_from_gdrive(GDRIVE_FILE_ID, MODEL_PATH)
    except Exception as e:
        sys.exit(f"[ERROR] Không download được model: {e}")

print(f"[INFO] Loading model...")
model   = load_model_safe(MODEL_PATH)
CLASSES = load_classes()
print(f"[INFO] ✅ Sẵn sàng! {len(CLASSES)} classes")


# ─────────────────────────────────────────────
# Preprocess
# ─────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, 0)


# ─────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────
app = FastAPI(title="Plant Disease API", version="3.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "running",
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "classes": len(CLASSES),
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "num_classes": len(CLASSES)}

@app.get("/classes")
def get_classes():
    return {"classes": CLASSES, "total": len(CLASSES)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Chỉ chấp nhận file ảnh")
    data = await file.read()
    if not data:
        raise HTTPException(400, "File ảnh rỗng")
    try:
        tensor = preprocess(data)
    except Exception as e:
        raise HTTPException(422, f"Không đọc được ảnh: {e}")
    preds = model.predict(tensor, verbose=0)[0]
    idx   = int(np.argmax(preds))
    return {
        "predicted_class": CLASSES[idx] if idx < len(CLASSES) else f"Class_{idx}",
        "confidence": round(float(preds[idx]) * 100, 2),
        "class_index": idx,
    }

@app.post("/predict/top5")
async def predict_top5(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Chỉ chấp nhận file ảnh")
    data = await file.read()
    if not data:
        raise HTTPException(400, "File ảnh rỗng")
    try:
        tensor = preprocess(data)
    except Exception as e:
        raise HTTPException(422, f"Không đọc được ảnh: {e}")
    preds    = model.predict(tensor, verbose=0)[0]
    top5_idx = np.argsort(preds)[::-1][:5]
    return {
        "top5": [
            {
                "rank": i + 1,
                "class": CLASSES[idx] if idx < len(CLASSES) else f"Class_{idx}",
                "confidence": round(float(preds[idx]) * 100, 2),
                "class_index": int(idx),
            }
            for i, idx in enumerate(top5_idx)
        ]
    }
