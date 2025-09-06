import os
import random
import shutil

# === CONFIGURATION ===
source_folder = os.path.expanduser("~/Downloads/homeobjects-3K/train/images")
destination_folder = os.path.expanduser("~/Downloads/Licenta_scripts/CalibrationDataSet/calibration_images_400")
N = 400  # number of images to copy (between 100 and 1000)

# === CREATE DESTINATION FOLDER IF IT DOESN'T EXIST ===
os.makedirs(destination_folder, exist_ok=True)

# === GET ALL IMAGE FILES ===
image_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(image_extensions)]

# === RANDOMLY SELECT N FILES ===
if N > len(image_files):
    raise ValueError(f"Requested {N} images, but only {len(image_files)} found.")

selected_images = random.sample(image_files, N)

# === COPY SELECTED FILES TO DESTINATION ===
for filename in selected_images:
    src_path = os.path.join(source_folder, filename)
    dst_path = os.path.join(destination_folder, filename)
    shutil.copyfile(src_path, dst_path)

print(f"Copied {len(selected_images)} images to '{destination_folder}'.")

