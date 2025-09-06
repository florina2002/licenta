#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import shutil
from xml.dom import minidom
import xml.etree.ElementTree as ET
from PIL import Image

# -------- Config (edit if your layout differs) --------
HOME = Path(os.environ.get("HOME", "~")).expanduser()
ROOT_YOLO = HOME / "kv260-yolo3" / "data" / "homeobjects-3K"   # expects images/{train,val}, labels/{train,val}
CLASSES_TXT = HOME / "kv260-yolo3" / "data" / "classes.txt"     # 12 lines, order matters
OUT_VOC = HOME / "kv260-yolo3" / "data" / "homeobjects-voc"
SPLITS = ["train", "val"]
IMG_EXTS = (".jpg", ".jpeg", ".png")
# ------------------------------------------------------

def ensure_dirs():
    (OUT_VOC / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (OUT_VOC / "Annotations").mkdir(parents=True, exist_ok=True)
    (OUT_VOC / "ImageSets" / "Main").mkdir(parents=True, exist_ok=True)

def read_classes(classes_txt: Path):
    if not classes_txt.exists():
        print(f"[ERROR] classes file not found: {classes_txt}", file=sys.stderr)
        sys.exit(1)
    classes = [c.strip() for c in classes_txt.read_text().splitlines() if c.strip()]
    if not classes:
        print("[ERROR] classes.txt is empty.", file=sys.stderr)
        sys.exit(1)
    return classes

def voc_xml(fname: str, width: int, height: int, objects):
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = "homeobjects-voc"
    ET.SubElement(ann, "filename").text = fname
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    for (cls, xmin, ymin, xmax, ymax) in objects:
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = cls
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(xmin)
        ET.SubElement(bb, "ymin").text = str(ymin)
        ET.SubElement(bb, "xmax").text = str(xmax)
        ET.SubElement(bb, "ymax").text = str(ymax)
    return minidom.parseString(ET.tostring(ann)).toprettyxml(indent="  ")

def convert_split(split: str, classes):
    images_dir = ROOT_YOLO / "images" / split
    labels_dir = ROOT_YOLO / "labels" / split
    if not images_dir.exists():
        print(f"[ERROR] Missing images dir: {images_dir}", file=sys.stderr)
        sys.exit(1)
    if not labels_dir.exists():
        print(f"[ERROR] Missing labels dir: {labels_dir}", file=sys.stderr)
        sys.exit(1)

    ids_path = OUT_VOC / "ImageSets" / "Main" / f"{split}.txt"
    with ids_path.open("w") as ids_f:
        img_count = 0
        obj_count = 0
        for img_p in sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]):
            try:
                with Image.open(img_p) as img:
                    w, h = img.size
            except Exception as e:
                print(f"[WARN] Skipping unreadable image {img_p}: {e}", file=sys.stderr)
                continue

            stem = img_p.stem
            label_p = labels_dir / f"{stem}.txt"
            objects = []

            if label_p.exists():
                for raw in label_p.read_text().splitlines():
                    parts = raw.strip().split()
                    if len(parts) != 5:
                        # skip empty/malformed lines
                        continue
                    try:
                        cid, xc, yc, bw, bh = map(float, parts)
                        cid = int(cid)
                        if cid < 0 or cid >= len(classes):
                            # skip bad class ids
                            continue
                        cls = classes[cid]
                        # YOLO (cx,cy,w,h) relative → VOC absolute corners
                        xmin = int((xc - bw / 2.0) * w)
                        ymin = int((yc - bh / 2.0) * h)
                        xmax = int((xc + bw / 2.0) * w)
                        ymax = int((yc + bh / 2.0) * h)
                        # clamp
                        xmin = max(1, min(xmin, w - 1))
                        ymin = max(1, min(ymin, h - 1))
                        xmax = max(1, min(xmax, w - 1))
                        ymax = max(1, min(ymax, h - 1))
                        # discard degenerate boxes
                        if xmax <= xmin or ymax <= ymin:
                            continue
                        objects.append((cls, xmin, ymin, xmax, ymax))
                    except Exception:
                        continue

            # copy image (normalize to .jpg extension in VOC folder)
            dst_img = OUT_VOC / "JPEGImages" / f"{stem}.jpg"
            if not dst_img.exists():
                try:
                    shutil.copy(img_p, dst_img)
                except Exception as e:
                    print(f"[WARN] Could not copy {img_p} → {dst_img}: {e}", file=sys.stderr)
                    continue

            # write xml
            xml_txt = voc_xml(f"{stem}.jpg", w, h, objects)
            (OUT_VOC / "Annotations" / f"{stem}.xml").write_text(xml_txt)

            # add to split listing
            ids_f.write(stem + "\n")

            img_count += 1
            obj_count += len(objects)

    print(f"[{split}] images: {img_count}, objects: {obj_count}, list: {ids_path}")

def main():
    ensure_dirs()
    classes = read_classes(CLASSES_TXT)
    print(f"Loaded {len(classes)} classes from {CLASSES_TXT}")
    for split in SPLITS:
        convert_split(split, classes)
    print("VOC dataset created at:", OUT_VOC)

if __name__ == "__main__":
    main()

