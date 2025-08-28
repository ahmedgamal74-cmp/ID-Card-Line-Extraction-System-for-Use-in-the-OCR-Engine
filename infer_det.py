# python infer_det.py --input_dir 2_IDs_derotated/ --save_dir 3_ROIs_det/

import argparse
from pathlib import Path
import os
import cv2
import torch
from ultralytics import YOLO
from config import *

# Configrations
if detection_model=='detr':
    MODEL_PATH = r"models\rt_detr_1.pt"  
else:
    MODEL_PATH = r"models\yolov8s_detect.pt"

# if torch.cuda.is_available():
#     DEVICE="cuda:0"
# else:
#     DEVICE="cpu"

device=device

CONF = 0.65
IOU = 0.7
IMG_SIZE = 640
image_types = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def clamp(val, lo, hi):
    return max(lo, min(int(val), hi))

def run_folder(input_dir: Path, save_dir: Path):

    # load detection model
    model = YOLO(MODEL_PATH)

    # read images from the folder
    imgs = [p for p in input_dir.rglob("*") if p.suffix.lower() in image_types]
    if not imgs:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(imgs)} images")
    print(f"Inference on {device} started...")

    # return results in stream
    count=0
    for r in model.predict(source=[str(p) for p in imgs], conf=CONF, iou=IOU, imgsz=IMG_SIZE,
                           device=device, verbose=False, stream=True):
        
        img_path = Path(getattr(r, "path", "image")).resolve()
        img_stem = img_path.stem

        # output folder for each image
        out_dir = save_dir / img_stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # save annotated image
        annotated = r.plot()
        cv2.imwrite(str(out_dir / f"{img_stem}_annotated.jpg"), annotated)

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

        count+=1
        print(f"Image {count}: {img_path.name}: {len(r.boxes)} detections saved to {out_dir}")

def parse_args():
    ap = argparse.ArgumentParser(description="YOLO inference on a floder")
    ap.add_argument("--input_dir", required=True, help="input images folder")
    ap.add_argument("--save_dir", required=True, help="output images folder")
    return ap.parse_args()

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    save_dir = Path(args.save_dir)
    # if not input_dir.exists():
    #     raise FileNotFoundError(f"input_dir not found: {input_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    run_folder(input_dir, save_dir)

if __name__ == "__main__":
    main()
