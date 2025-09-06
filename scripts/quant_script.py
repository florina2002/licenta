#!/usr/bin/env python3
import os, math, glob
import tensorflow as tf

# --- Paths ---
FLOAT_MODEL_PATH = "/workspace/kv260-yolo3/exports/altele/yolov3_homeobjects_float.h5"
CALIB_DIR        = "/workspace/kv260-yolo3/calib_images_1"
IMG_SIZE         = 416
BATCH_SIZE       = 1   # concrete batch; keeps memory predictable
OUTPUT_DIR       = "quantized_exports"
OUTPUT_NAME      = "yolov3_homeobjects_quant.h5"

# GPU memory growth (safe no-op if no GPU)
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass

os.makedirs(OUTPUT_DIR, exist_ok=True)

def list_images(folder):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)

def load_and_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img

def build_calib_dataset(files, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)

def main():
    img_files = list_images(CALIB_DIR)
    if not img_files:
        raise FileNotFoundError(f"No images found in '{CALIB_DIR}'")

    calib_ds = build_calib_dataset(img_files, BATCH_SIZE)
    calib_steps = math.ceil(len(img_files) / BATCH_SIZE)
    print(f"[INFO] Found {len(img_files)} calibration images.")
    print(f"[INFO] Calibration steps computed: {calib_steps}")

    # --- Import YOLO code BEFORE loading, so Lambda layers resolve ---
    # If these imports don't exist in your environment, adjust as needed:
    import yolov3_tf2.models as yolo_models
    from yolov3_tf2.models import YoloV3

    custom_objects = {}  # add any custom ops if your model uses them

    print(f"[INFO] Loading float model: {FLOAT_MODEL_PATH}")
    float_model = tf.keras.models.load_model(
        FLOAT_MODEL_PATH,
        compile=False,
        custom_objects=custom_objects
    )

    # Warm-up (ensures graph built with concrete shape)
    _ = float_model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3), tf.float32))

    print("[INFO] Creating VitisQuantizer...")
    from tensorflow_model_optimization.quantization.keras import vitis_quantize
    quantizer = vitis_quantize.VitisQuantizer(float_model)

    print("[INFO] Starting PTQ (no FastFT, CLE on)...")
    quantized_model = quantizer.quantize_model(
        calib_dataset=calib_ds,
        calib_steps=calib_steps,
        # Use a concrete batch dimension instead of None:
        input_shape=[1, IMG_SIZE, IMG_SIZE, 3],
        include_cle=True,
        include_fast_ft=False
    )

    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    quantized_model.save(out_path, include_optimizer=False)
    print(f"[SUCCESS] Saved quantized model to: {out_path}")

if __name__ == "__main__":
    main()

