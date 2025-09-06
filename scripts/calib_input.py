import os, glob, cv2, numpy as np, tensorflow as tf

# Point to your calib images dir
CALIB_DIR = "/workspace/kv260-yolo3/calib_images"
SIZE = (416, 416)

def _load(path):
    img = cv2.imread(path)[:,:,::-1]           # BGR->RGB
    img = cv2.resize(img, SIZE).astype("float32")/255.0
    return img

def calib_input(iter=200):
    files = sorted(glob.glob(os.path.join(CALIB_DIR, "*.jpg")))
    if not files:
        raise RuntimeError(f"No .jpg images found in {CALIB_DIR}")
    files = files[:min(iter, len(files))]

    def gen():
        for p in files:
            img = _load(p)
            yield (img,)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(tf.TensorSpec(shape=(416,416,3), dtype=tf.float32),)
    )
    ds = ds.batch(1)
    return ds
