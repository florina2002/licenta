#!/usr/bin/env python3
"""
Create TFRecord files from a VOC-style dataset generated from HomeObjects-3K.

Defaults assume:
  ROOT/
    Annotations/*.xml
    JPEGImages/*.jpg|.png
    ImageSets/Main/train.txt
    ImageSets/Main/val.txt
and classes listed (one per line) in classes.txt (IDs = line index + 1).

Usage (default paths for your setup):
  python voc_to_tfrecord.py

Or custom:
  python voc_to_tfrecord.py \
    --root /home/florina/kv260-yolo3/data/homeobjects-voc \
    --classes /home/florina/kv260-yolo3/data/classes.txt \
    --out-train /home/florina/kv260-yolo3/data/homeobjects_train.record \
    --out-val   /home/florina/kv260-yolo3/data/homeobjects_val.record \
    --strict
"""
import os
import io
import sys
import argparse
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import tensorflow as tf
from PIL import Image

# ---------- Helpers ----------
def _norm(s: str) -> str:
    """Normalize class names to avoid tiny mismatches."""
    return " ".join(s.strip().lower().replace("_", " ").split())

def load_name_to_id_from_classes_txt(p: Path) -> Dict[str, int]:
    if not p.exists():
        raise FileNotFoundError(f"classes.txt not found: {p}")
    raw = [c.strip() for c in p.read_text().splitlines() if c.strip()]
    if not raw:
        raise RuntimeError("classes.txt is empty.")
    # IDs are 1..N to match typical pbtxt maps
    return {_norm(name): i + 1 for i, name in enumerate(raw)}

def parse_xml(ann_path: Path) -> Tuple[str, int, int, List[Tuple[str, int, int, int, int]]]:
    """Return: (filename, width, height, [(class_name, xmin, ymin, xmax, ymax), ...])."""
    tree = ET.parse(str(ann_path))
    root = tree.getroot()
    fname = root.findtext("filename")
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    objs = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bnd = obj.find("bndbox")
        xmin = int(float(bnd.findtext("xmin")))
        ymin = int(float(bnd.findtext("ymin")))
        xmax = int(float(bnd.findtext("xmax")))
        ymax = int(float(bnd.findtext("ymax")))
        # clamp & discard degenerate
        xmin = max(1, min(xmin, width - 1))
        ymin = max(1, min(ymin, height - 1))
        xmax = max(1, min(xmax, width - 1))
        ymax = max(1, min(ymax, height - 1))
        if xmax > xmin and ymax > ymin:
            objs.append((name, xmin, ymin, xmax, ymax))
    return fname, width, height, objs

# TF Feature builders
def _bytes_feature(v: bytes): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
def _bytes_list_feature(v: List[bytes]): return tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
def _float_list_feature(v: List[float]): return tf.train.Feature(float_list=tf.train.FloatList(value=v))
def _int64_list_feature(v: List[int]): return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

def example_from_record(img_dir: Path, ann_dir: Path, item: str, name_to_id: Dict[str, int], strict: bool) -> tf.train.Example:
    ann_path = ann_dir / f"{item}.xml"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotation: {ann_path}")
    fname, width, height, objs = parse_xml(ann_path)

    # Find image by the name in XML; fallback to stem.jpg
    img_path = img_dir / fname if fname else img_dir / f"{item}.jpg"
    if not img_path.exists():
        alt = img_dir / f"{item}.jpg"
        if alt.exists():
            img_path = alt
        else:
            raise FileNotFoundError(f"Missing image for {item}: tried {img_path} and {alt}")

    with tf.io.gfile.GFile(str(img_path), "rb") as fid:
        encoded = fid.read()

    # Ensure JPEG bytes; convert if needed (e.g., PNG)
    try:
        Image.open(io.BytesIO(encoded)).verify()
        img_format = b"jpeg" if img_path.suffix.lower() in [".jpg", ".jpeg"] else b"png"
        if img_format != b"jpeg":
            raise Exception("Converting to JPEG")
    except Exception:
        pil = Image.open(str(img_path)).convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=95)
        encoded = buf.getvalue()

    key = hashlib.sha256(encoded).hexdigest()

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text: List[bytes] = []
    classes: List[int] = []

    for (name, xmin, ymin, xmax, ymax) in objs:
        k = _norm(name)
        if k not in name_to_id:
            msg = f"Class '{name}' not in label map (normalized='{k}')."
            if strict:
                raise KeyError(msg)
            else:
                # Skip unknown class
                print(f"[WARN] {item}: {msg}", file=sys.stderr)
                continue
        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)
        classes_text.append(name.encode("utf8"))  # raw, not normalized, for readability
        classes.append(int(name_to_id[k]))

    features = tf.train.Features(feature={
        "image/height": _int64_list_feature([height]),
        "image/width":  _int64_list_feature([width]),
        "image/filename": _bytes_feature((fname or f"{item}.jpg").encode("utf8")),
        "image/source_id": _bytes_feature((fname or f"{item}.jpg").encode("utf8")),
        "image/encoded": _bytes_feature(encoded),
        "image/format": _bytes_feature(b"jpeg"),
        "image/key/sha256": _bytes_feature(key.encode("utf8")),
        "image/object/bbox/xmin": _float_list_feature(xmins),
        "image/object/bbox/xmax": _float_list_feature(xmaxs),
        "image/object/bbox/ymin": _float_list_feature(ymins),
        "image/object/bbox/ymax": _float_list_feature(ymaxs),
        "image/object/class/text": _bytes_list_feature(classes_text),
        "image/object/class/label": _int64_list_feature(classes),
    })
    return tf.train.Example(features=features)

def write_split(root: Path, split: str, out_path: Path, name_to_id: Dict[str, int], strict: bool):
    img_dir = root / "JPEGImages"
    ann_dir = root / "Annotations"
    list_path = root / "ImageSets" / "Main" / f"{split}.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"Missing list file: {list_path}")
    ids = [x.strip() for x in list_path.read_text().splitlines() if x.strip()]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count, boxes = 0, 0
    with tf.io.TFRecordWriter(str(out_path)) as w:
        for item in ids:
            try:
                ex = example_from_record(img_dir, ann_dir, item, name_to_id, strict)
                # count boxes by re-parsing (cheap)
                _, _, _, objs = parse_xml(ann_dir / f"{item}.xml")
                boxes += len(objs)
                w.write(ex.SerializeToString())
                count += 1
            except Exception as e:
                print(f"[WARN] Skipping {item}: {e}", file=sys.stderr)
    print(f"[{split}] wrote {count} examples ({boxes} boxes) -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/home/florina/kv260-yolo3/data/homeobjects-voc"))
    ap.add_argument("--classes", type=Path, default=Path("/home/florina/kv260-yolo3/data/classes.txt"))
    ap.add_argument("--out-train", type=Path, default=Path("/home/florina/kv260-yolo3/data/homeobjects_train.record"))
    ap.add_argument("--out-val",   type=Path, default=Path("/home/florina/kv260-yolo3/data/homeobjects_val.record"))
    ap.add_argument("--strict", action="store_true", help="Error on unknown class names (default: skip).")
    args = ap.parse_args()

    name_to_id = load_name_to_id_from_classes_txt(args.classes)
    print(f"Loaded {len(name_to_id)} classes from {args.classes}")
    write_split(args.root, "train", args.out_train, name_to_id, args.strict)
    write_split(args.root, "val",   args.out_val,   name_to_id, args.strict)
    print("Done.")

if __name__ == "__main__":
    main()

