#!/usr/bin/env python3
import os, sys, json, argparse, pathlib
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def add_repo_to_path():
    candidates = [
        "/workspace/kv260-yolo3/src/yolov3-tf2",
        os.path.join(os.path.expanduser("~"), "kv260-yolo3", "src", "yolov3-tf2"),
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "yolov3_tf2")) and c not in sys.path:
            sys.path.insert(0, c)
            return c
    return None

def parse_args():
    ap = argparse.ArgumentParser("Export YOLOv3 SavedModel and freeze to TF1 .pb")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .tf")
    ap.add_argument("--export_dir", required=True, help="SavedModel output dir")
    ap.add_argument("--frozen_pb", required=True, help="Frozen GraphDef .pb output path")
    ap.add_argument("--img_size", type=int, default=416)
    ap.add_argument("--classes", type=int, default=12)
    ap.add_argument("--calib_dir", default="/workspace/home_objects_200_images", help="For helper scripts")
    return ap.parse_args()

def main():
    args = parse_args()
    repo = add_repo_to_path()
    from yolov3_tf2.models import YoloV3

    for g in tf.config.list_physical_devices('GPU'):
        try: tf.config.experimental.set_memory_growth(g, True)
        except Exception: pass

    IMG, CLASSES = args.img_size, args.classes

    print(f"[INFO] Using repo path: {repo}")
    print(f"[INFO] Building YoloV3(size={IMG}, classes={CLASSES}) and loading: {args.ckpt}")

