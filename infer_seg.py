# python infer_seg.py --input_dir 2_IDs_derotated/ --save_dir 3_ROIs_seg/

import argparse
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from config import *

MODEL_PATH = r"models\yolov8s_seg.pt"
CONF = 0.25
IMGSZ = 640
device=device
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def clamp(v, lo, hi): return max(lo, min(int(v), hi))

def run_folder(input_dir: Path, save_dir: Path):
    # # # pick device
    # # device_arg = DEVICE
    # if device != "cpu" and not torch.cuda.is_available():
    #     print("[WARN] CUDA not available, using CPU.")
    #     device_arg = "cpu"

    # load model once (seg or detect weights both fine)
    model = YOLO(MODEL_PATH)

    # gather images (recursive)
    imgs = [p for p in input_dir.rglob("*") if p.suffix.lower() in VALID_EXTS]
    if not imgs:
        print(f"[INFO] No images found in: {input_dir}")
        return

    print(f"[INFO] Found {len(imgs)} images. Running inference on device={device}...")

    # stream=True yields results one-by-one and includes r.path
    for r in model.predict(
        source=[str(p) for p in imgs],
        conf=CONF,
        imgsz=IMGSZ,
        device=device,
        verbose=False,
        stream=True
    ):
        img_path = Path(getattr(r, "path", "image")).resolve()
        img_stem = img_path.stem

        # per-image output folder
        out_dir = save_dir / img_stem
        out_dir.mkdir(parents=True, exist_ok=True)

        H, W = r.orig_img.shape[:2]
        names = r.names if hasattr(r, "names") else model.names

        # If no detections: save original as annotated (no masks), mark file, continue
        if r.boxes is None or len(r.boxes) == 0:
            cv2.imwrite(str(out_dir / f"{img_stem}_annotated.jpg"), r.orig_img)
            (out_dir / "_no_detections.txt").write_text("No detections.")
            print(f"[OK] {img_path.name}: 0 detections")
            continue

        # Build our own annotated image with RECTANGLES ONLY (no mask overlay)
        annotated = r.orig_img.copy()

        for i, box in enumerate(r.boxes):
            # xyxy -> ints within image bounds
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = clamp(x1, 0, W - 1)
            y1 = clamp(y1, 0, H - 1)
            x2 = clamp(x2, 0, W - 1)
            y2 = clamp(y2, 0, H - 1)
            if x2 <= x1 or y2 <= y1:
                continue

            # draw rectangle (no masks)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # label
            cls_id = int(box.cls.item()) if box.cls is not None else -1
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            conf = float(box.conf.item()) if box.conf is not None else 0.0
            label = f"{cls_name} {conf:.2f}"

            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_text = max(y1, th + 3)
            cv2.rectangle(annotated, (x1, y_text - th - 3), (x1 + tw + 2, y_text + baseline), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1 + 1, y_text - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # rectangular crop only (no alpha/mask files)
            crop = r.orig_img[y1:y2, x1:x2]
            cv2.imwrite(str(out_dir / f"{i:03d}_{cls_name}_{conf:.2f}.jpg"), crop)

        # save our box-only annotated image
        annotated_to_save = r.plot()
        cv2.imwrite(str(out_dir / f"{img_stem}_annotated.jpg"), annotated_to_save)
        print(f"[OK] {img_path.name}: {len(r.boxes)} detections -> {out_dir}")


def parse_args():
    ap = argparse.ArgumentParser(description="YOLO segmentation inference: save annotated image, masks, and alpha crops.")
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
