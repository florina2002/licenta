# image_calibration_reader.py (letterbox variant)
from PIL import Image
import numpy as np, os

def letterbox(img, new_size=640, color=(114,114,114)):
    w, h = img.size
    r = min(new_size / w, new_size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    img = img.resize((nw, nh), Image.BILINEAR)
    new_img = Image.new('RGB', (new_size, new_size), color)
    left = (new_size - nw) // 2
    top  = (new_size - nh) // 2
    new_img.paste(img, (left, top))
    return new_img

class ImageCalibrationDataReader:
    def __init__(self, image_dir, input_name='images', image_size=(640,640), max_images=None):
        paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                 if f.lower().endswith(('.jpg','.png','.jpeg'))]
        if max_images: paths = paths[:max_images]
        self.samples = []
        for p in paths:
            img = Image.open(p).convert('RGB')
            img = letterbox(img, new_size=image_size[0])
            x = np.array(img).astype(np.float32) / 255.0
            x = x.transpose(2,0,1)[None, ...]  # NCHW
            self.samples.append({input_name: x})
        self.i = 0
    def get_next(self):
        if self.i < len(self.samples):
            s = self.samples[self.i]; self.i += 1; return s
        return None
    def rewind(self): self.i = 0


