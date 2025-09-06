# /workspace/calib_input_fn.py
import os, glob, tensorflow as tf

IMG = 416
CALIB_DIR = os.environ.get("CALIB_DIR", "/workspace/home_objects_200_images")

def _pp(path):
    b = tf.io.read_file(path)
    x = tf.image.decode_image(b, channels=3, expand_animations=False)
    x = tf.image.convert_image_dtype(x, tf.float32)
    x = tf.image.resize(x, [IMG, IMG])
    x.set_shape([IMG, IMG, 3])
    return x

def input_fn():
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp")
    files = []
    for e in exts: files += glob.glob(os.path.join(CALIB_DIR, e))
    files = sorted(files)[:128]  # use up to 128 images for quicker PTQ
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.map(_pp, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(1)
    # IMPORTANT: return a dataset whose elements are **lists/tuples** of inputs
    return ds.map(lambda x: (x,))

