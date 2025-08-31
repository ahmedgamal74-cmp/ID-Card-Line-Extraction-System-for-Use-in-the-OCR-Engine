# python roi_extract.py 2_IDs_derotated/  ROIs.json --outdir 3_ROIs_classic 

import os
import cv2
import json, argparse, shutil
from pathlib import Path
from config import *

# config
image_types = image_input_types

"""
This function to make the output crops are withing the input image size
"""
def clamp(val, lo, hi): return max(lo, min((val), hi))

"""
extract and save the ROI crops from an image
"""
def process_one_image(image_path, rois, outdir):
    # read image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    if Path(outdir).exists() and Path(outdir).is_dir(): shutil.rmtree(Path(outdir))
    os.makedirs(outdir, exist_ok=True)

    crops_count = 0
    # loop on ROIs to extract
    for roi in rois:
        name = str(roi["name"]).strip()
        x1, y1, x2, y2 = int(roi["x1"]), int(roi["y1"]), int(roi["x2"]), int(roi["y2"])

        # sort and clamp within image bounds
        x_lo, x_hi = sorted((x1, x2))
        y_lo, y_hi = sorted((y1, y2))
        x_lo = clamp(x_lo, 0, w)
        x_hi = clamp(x_hi, 0, w)
        y_lo = clamp(y_lo, 0, h)
        y_hi = clamp(y_hi, 0, h)

        # crop to save
        crops_count+=1
        crop = img[y_lo:y_hi, x_lo:x_hi]
        out_path = os.path.join(outdir, f"{crops_count}_{name}.jpg")

        cv2.imwrite(out_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])        # 95% quality

    return crops_count

"""
main function
"""
def main():
    p = argparse.ArgumentParser()
    p.add_argument("image_path")
    p.add_argument("rois_json")
    p.add_argument("--outdir", default="roi_crops")
    args = p.parse_args()

    input_path = Path(args.image_path)

    # read ROIs from the json file
    with open(args.rois_json, "r", encoding="utf-8") as f: rois = json.load(f)

    image_conut = 0
    total_expected = 0
    # process and save each image
    images = [p for p in sorted(input_path.glob("*")) if p.suffix.lower() in image_types]
    for img_path in images:
        image_conut+=1
        out_image_path = Path(args.outdir) / img_path.stem
        num_crops = process_one_image(str(img_path), rois, str(out_image_path))
        print(f"For image ({image_conut}){Path(img_path).name}: saved {num_crops} crops  in {out_image_path}")

if __name__ == "__main__":
    main()