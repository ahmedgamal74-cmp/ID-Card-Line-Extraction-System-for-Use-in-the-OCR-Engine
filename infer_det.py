# usage: python infer_det.py --input_dir 2_IDs_derotated/ --save_dir 3_ROIs_det/

import cv2
import argparse, shutil
from pathlib import Path
from ultralytics import YOLO
from config import *

# Configrations
conf_thresh = 0.1
iou_thresh = 0.5
img_size = 640
device=device
image_types = image_input_types
model_path = detection_model_path  

"""
This function to make the output crops are withing the input image size
"""
def clamp(val, lo, hi): return max(lo, min(int(val), hi))

"""
This function takes two "Path" types to run model inference on the input directory and save crops in another one
"""
def run_inference(input_dir, save_dir):

    # load detection model
    model = YOLO(model_path)

    # read images from the folder
    imgs = [p for p in input_dir.rglob("*") if p.suffix.lower() in image_types]
    if not imgs:
        print(f"NO images found in {input_dir}")
        return

    print(f"Found {len(imgs)} images")
    print(f"Starting inference on {device}...")

    img_count=0
    # get model results one by one from images list
    for r in model.predict(source=[str(p) for p in imgs], conf=conf_thresh, iou=iou_thresh, imgsz=img_size,
                           device=device, verbose=False, stream=True):
        # set the image output path
        img_path = Path(getattr(r, "path", "image")).resolve()
        img_stem = img_path.stem
        out_dir = save_dir / img_stem
        if Path(out_dir).exists() and Path(out_dir).is_dir(): shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        H, W = r.orig_img.shape[:2]
        names = r.names if hasattr(r, "names") else model.names

        det_count=0
        # looping on detections to search for out ROIs
        for i, box in enumerate(r.boxes):
            # crop within image bounds
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = clamp(x1, 0, W - 1)
            y1 = clamp(y1, 0, H - 1)
            x2 = clamp(x2, 0, W - 1)
            y2 = clamp(y2, 0, H - 1)

            # set each crop label
            class_id = int(box.cls.item()) if box.cls is not None else -1
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            conf = float(box.conf.item()) if box.conf is not None else 0.0
            crop = r.orig_img[y1:y2, x1:x2]

            # checks to save only the four ROI crops
            if det_count>=4: break
            if class_name == "firstname":
                crop_num = 1
            elif class_name == "second name":
                crop_num = 2
                class_name = "second_name"
            elif class_name == "location":
                crop_num = 3 
            elif class_name == "national_id":
                crop_num = 4 
            else:
                continue  
            crop_name = f"{crop_num}_{class_name}_{conf:.2f}.jpg"
            cv2.imwrite(str(out_dir / crop_name), crop)
            det_count+=1

        img_count+=1
        print(f"For image ({img_count}){img_path.name}: saved {det_count} crops in {out_dir}")

"""
This function parses the arguments of the sccript
"""
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    return ap.parse_args()

"""
main function
"""
def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    save_dir = Path(args.save_dir)
    if not input_dir.exists(): raise FileNotFoundError(f"Input directory {input_dir} not found")
    if save_dir.exists() and save_dir.is_dir(): shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_inference(input_dir, save_dir)

if __name__ == "__main__":
    main()