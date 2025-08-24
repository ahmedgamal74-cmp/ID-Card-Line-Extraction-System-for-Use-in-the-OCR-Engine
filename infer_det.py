# python infer_det.py --input_dir 2_IDs_derotated/ --save_dir 3_ROIs_det/

import argparse
from pathlib import Path
import os
import cv2
import torch
from ultralytics import YOLO

# --- config (edit MODEL_PATH) ---
MODEL_PATH = r"models\rt_detr_1.pt"  # yolov8s_detect   /    rt_detr_1
CONF = 0.65
IOU = 0.7
IMGSZ = 640
DEVICE = 0  # GPU 0; will fall back to CPU if CUDA isn't available

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def clamp(val, lo, hi):
    return max(lo, min(int(val), hi))

def run_folder(input_dir: Path, save_dir: Path):
    # pick device
    device_arg = DEVICE
    if device_arg != "cpu" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU.")
        device_arg = "cpu"

    # load model once
    model = YOLO(MODEL_PATH)

    # gather images (recursive)
    imgs = [p for p in input_dir.rglob("*") if p.suffix.lower() in VALID_EXTS]
    if not imgs:
        print(f"[INFO] No images found in: {input_dir}")
        return

    print(f"[INFO] Found {len(imgs)} images. Running inference on device={device_arg}...")

    # stream=True yields results one-by-one and includes r.path
    for r in model.predict(
        source=[str(p) for p in imgs],
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        device=device_arg,
        verbose=False,
        stream=True
    ):
        img_path = Path(getattr(r, "path", "image")).resolve()
        img_stem = img_path.stem

        # per-image output folder
        out_dir = save_dir / img_stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # save annotated image
        annotated = r.plot()
        cv2.imwrite(str(out_dir / f"{img_stem}_annotated.jpg"), annotated)

        # extract and save crops for each detection
        if r.boxes is None or len(r.boxes) == 0:
            # still drop an empty marker for traceability
            (out_dir / "_no_detections.txt").write_text("No detections.")
            print(f"[OK] {img_path.name}: 0 detections")
            continue

        H, W = r.orig_img.shape[:2]
        names = r.names if hasattr(r, "names") else model.names

        for i, box in enumerate(r.boxes):
            # xyxy -> ints within image bounds
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = clamp(x1, 0, W - 1)
            y1 = clamp(y1, 0, H - 1)
            x2 = clamp(x2, 0, W - 1)
            y2 = clamp(y2, 0, H - 1)

            if x2 <= x1 or y2 <= y1:
                # skip degenerate boxes
                continue

            crop = r.orig_img[y1:y2, x1:x2]

            cls_id = int(box.cls.item()) if box.cls is not None else -1
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            conf = float(box.conf.item()) if box.conf is not None else 0.0

            crop_name = f"{i:03d}_{cls_name}_{conf:.2f}.jpg"
            cv2.imwrite(str(out_dir / crop_name), crop)

        print(f"[OK] {img_path.name}: {len(r.boxes)} detections -> {out_dir}")

def parse_args():
    ap = argparse.ArgumentParser(
        description="Run YOLO on a folder, save per-image detection crops and annotated images."
    )
    ap.add_argument("--input_dir", required=True, help="Folder of input images")
    ap.add_argument("--save_dir", required=True, help="Where to save results")
    return ap.parse_args()

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    save_dir = Path(args.save_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    run_folder(input_dir, save_dir)

if __name__ == "__main__":
    main()
