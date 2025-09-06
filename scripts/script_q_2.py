#!/usr/bin/env python3
# Quantize YOLOv3 for Vitis AI from a TF checkpoint (KV260-friendly).
# - Rebuilds model from code (no H5 lambdas)
# - Wraps outputs into a list (avoids: "[VAI ERROR] output_layer format is not dict or list.")
# - Uses CLE, disables FastFT (CPU-safe)
import os, sys, math, glob
import tensorflow as tf

# --------------------------- Config ---------------------------
# Edit these to match your paths. You can also override via env vars.
CKPT = os.environ.get(
    "YOLO_CKPT",
    "/workspace/kv260-yolo3/src/yolov3-tf2/checkpoints/yolov3_train_7.tf"
)
CALIB_DIR = os.environ.get("CALIB_DIR", "/workspace/kv260-yolo3/calib_images2")  # relative OK
IMG_SIZE = int(os.environ.get("IMG_SIZE", "416"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))  # keep small on CPU
OUT_DIR = os.environ.get(
    "OUT_DIR",
    "/workspace/kv260-yolo3/src/yolov3-tf2/quantized_exports"
)
OUT_NAME = os.environ.get("OUT_NAME", "yolov3_homeobjects_quant.h5")
INCLUDE_CLE = os.environ.get("INCLUDE_CLE", "1") not in ("0", "false", "False")
INCLUDE_FAST_FT = False  # keep off unless you truly need it
# -------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# Make sure the yolov3_tf2 package is importable (repo uses hyphen in folder name)
candidate_paths = [
    "/workspace/kv260-yolo3/src/yolov3-tf2",
    os.path.join(os.path.expanduser("~"), "kv260-yolo3", "src", "yolov3-tf2"),
    os.getcwd(),
    os.path.dirname(os.path.abspath(__file__)),
]
for p in candidate_paths:
    if os.path.isdir(os.path.join(p, "yolov3_tf2")) and p not in sys.path:
        sys.path.insert(0, p)

from yolov3_tf2.models import YoloV3
from tensorflow_model_optimization.quantization.keras import vitis_quantize

# Be gentle with GPU memory if any (no-op on CPU)
for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

def list_images(folder):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)

def preprocess(path):
    b = tf.io.read_file(path)
    x = tf.image.decode_image(b, channels=3, expand_animations=False)
    x = tf.image.convert_image_dtype(x, tf.float32)   # [0,1]
    x = tf.image.resize(x, [IMG_SIZE, IMG_SIZE])
    x.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return x

def build_ds(paths, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)

def main():
    # Resolve calibration directory (absolute or relative)
    calib_dir_abs = CALIB_DIR if os.path.isabs(CALIB_DIR) else os.path.join(os.getcwd(), CALIB_DIR)
    imgs = list_images(calib_dir_abs)
    if not imgs:
        raise SystemExit(f"[ERROR] No images found in '{calib_dir_abs}'")

    ds = build_ds(imgs, BATCH_SIZE)
    steps = math.ceil(len(imgs) / BATCH_SIZE)
    print(f"[INFO] Calib images: {len(imgs)}  steps: {steps}")

    # Rebuild YOLOv3 (training=True -> raw heads, no NMS/decoding lambdas)
    base = YoloV3(classes=12, size=IMG_SIZE, training=True)
    base.load_weights(CKPT).expect_partial()
    # Warm-up to build shapes concretely
    _ = base(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3), tf.float32))

    # ---- Wrap outputs to list/dict so Vitis quantizer doesn't error on tuples ----
    outs = base.outputs
    if isinstance(outs, tuple):
        outs = list(outs)
    elif not isinstance(outs, (list, dict)):
        outs = [outs]
    model = tf.keras.Model(inputs=base.inputs, outputs=outs, name="yolov3_for_quant")
    # -----------------------------------------------------------------------------

    print("[INFO] Creating VitisQuantizer...")
    q = vitis_quantize.VitisQuantizer(model)

    print(f"[INFO] Starting PTQ (CLE={INCLUDE_CLE}, FastFT={INCLUDE_FAST_FT})...")
    quant = q.quantize_model(
        calib_dataset=ds,
        calib_steps=steps,
        input_shape=[1, IMG_SIZE, IMG_SIZE, 3],   # concrete batch dim
        include_cle=INCLUDE_CLE,
        include_fast_ft=INCLUDE_FAST_FT,
    )

    out_path = os.path.join(OUT_DIR, OUT_NAME)
    quant.save(out_path, include_optimizer=False)
    print(f"[SUCCESS] Saved quantized model to: {out_path}")

if __name__ == "__main__":
    main()

