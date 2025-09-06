#!/usr/bin/env python3
import os, sys, argparse, pathlib, tensorflow as tf

def add_repo_to_path():
    # Common locations for the repo; add the one that contains `yolov3_tf2/`
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
    ap = argparse.ArgumentParser(description="Export YOLOv3 SavedModel from TF checkpoint")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .tf (no NMS)")
    ap.add_argument("--export_dir", required=True, help="Output SavedModel directory")
    ap.add_argument("--img_size", type=int, default=416, help="Input size (square)")
    ap.add_argument("--classes", type=int, default=12, help="Number of classes")
    return ap.parse_args()

def main():
    args = parse_args()

    repo_used = add_repo_to_path()
    if not repo_used:
        print("[WARN] Could not auto-locate yolov3_tf2 in sys.path; assuming it is installed.")

    # Gentle with any available GPU (no-op on CPU)
    for g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    # Import after sys.path is set
    from yolov3_tf2.models import YoloV3

    IMG = args.img_size
    CLASSES = args.classes
    CKPT = args.ckpt
    EXPORT = args.export_dir

    os.makedirs(EXPORT, exist_ok=True)

    print(f"[INFO] Using repo path: {repo_used}")
    print(f"[INFO] Building YOLOv3(size={IMG}, classes={CLASSES})")
    model = YoloV3(classes=CLASSES, size=IMG, training=True)

    print(f"[INFO] Loading weights from checkpoint: {CKPT}")
    model.load_weights(CKPT).expect_partial()

    # Warm-up to materialize shapes
    _ = model(tf.zeros((1, IMG, IMG, 3), tf.float32))

    # Define a clean serving signature:
    #  - single named input "images"
    #  - outputs as a dict (y1, y2, y3) to avoid tuple serialization quirks
    @tf.autograph.experimental.do_not_convert
    @tf.function(input_signature=[
        tf.TensorSpec([None, IMG, IMG, 3], tf.float32, name="images")
    ])
    def serving(x):
        y = model(x, training=False)
        if isinstance(y, (list, tuple)):
            y = {f"y{i+1}": y[i] for i in range(len(y))}
        return y

    print(f"[INFO] Exporting SavedModel to: {EXPORT}")
    tf.saved_model.save(model, EXPORT, signatures={'serving_default': serving})

    # Small summary of what got saved
    print("\n[SUCCESS] SavedModel exported.")
    print(f"Location: {EXPORT}")
    print("Contents:")
    for p in sorted(pathlib.Path(EXPORT).rglob("*")):
        rel = p.relative_to(EXPORT)
        if rel.parts[0] in (".git", "__pycache__"):  # skip noise
            continue
        print("  ", rel)

    # Print input/output names to help with next steps (freeze/quantize)
    loaded = tf.saved_model.load(EXPORT)
    sig = loaded.signatures["serving_default"]
    in_names = [t.name for t in sig.inputs]
    out_names = [t.name for t in sig.outputs]
    print("\n[INFO] Signature I/O (useful later):")
    print("  INPUTS :", in_names)
    print("  OUTPUTS:", out_names)
    print("\nTip: Next, freeze and print exact TF1 node names with:")
    print("  python - <<'PY'\n"
          "from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2\n"
          "import tensorflow as tf, json, os\n"
          f"loaded=tf.saved_model.load(r'{EXPORT}'); f=loaded.signatures['serving_default']\n"
          "frozen=convert_variables_to_constants_v2(f)\n"
          "gd=frozen.graph.as_graph_def(); out_dir='/workspace/frozen'; os.makedirs(out_dir, exist_ok=True)\n"
          "tf.io.write_graph(gd,out_dir,'yolov3_frozen.pb',as_text=False)\n"
          "print('INPUTS:',[t.name for t in frozen.inputs])\n"
          "print('OUTPUTS:',[t.name for t in frozen.outputs])\n"
          "open(os.path.join(out_dir,'io_names.json'),'w').write(json.dumps({'inputs':[t.name for t in frozen.inputs],'outputs':[t.name for t in frozen.outputs]},indent=2))\n"
          "print('Saved:', os.path.join(out_dir,'yolov3_frozen.pb'))\n"
          "PY")

if __name__ == "__main__":
    main()

