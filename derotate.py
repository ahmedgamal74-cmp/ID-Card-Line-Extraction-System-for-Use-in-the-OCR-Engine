# usage: python derotate.py --ref_dir 0_IDs_derotation_ref/ --img_dir 1_IDs_rectified/  --save_dir 2_IDs_derotated/

# correlation on a fixed area with respect to a set of reeference image to enhance/generalize correlation
# if correlation < threshold then the image need 180 degree rotation

import cv2
import argparse, shutil
from pathlib import Path
import numpy as np
from config import *

# config
# coorelation area -> the green area of the ID because it is distinct
corr_roi = (416, 3, 908, 156)    # (x1, y1, x2, y2)
corr_thresh = 0.2
image_types = image_input_types

"""
crop ROI within image bounds
"""
def clamp_roi(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    return x1, y1, x2, y2

"""
this function calcualte the pearson correalation after normalization
+1 means +ve correlation
~0 means no correlation = needs derotation
-1 means -ve correlation
"""
def pearson_corr(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    return float((a * b).mean() / (a.std() * b.std() + 1e-6))

"""
normalized cross correlation on the green area
"""
def corr_green(roi_ref, roi_in):
    return pearson_corr(roi_ref[..., 1], roi_in[..., 1])

"""
this function averages the reference ROI by pixel to get a generalized reference
"""
def average_ref_roi(ref_imgs, roi):
    acc = None
    count = 0
    x1 = y1 = x2 = y2 = None
    # acculmlate the ROI values of ref images values 
    for _, ref_img in ref_imgs:
        h, w = ref_img.shape[:2]
        x1, y1, x2, y2 = clamp_roi(*roi, w, h)
        roi_crop = ref_img[y1:y2, x1:x2].astype(np.float32)
        if acc is None:
            acc = np.zeros_like(roi_crop, dtype=np.float32)
        acc += roi_crop
        count += 1
    # calculate and return the avg    
    mean_roi = (acc / max(count, 1)).astype(np.uint8)
    return mean_roi

"""
main function
"""
def main():
    # argument pareser
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    args = ap.parse_args()
    ref_dir = Path(args.ref_dir)
    img_dir = Path(args.img_dir)
    save_dir = Path(args.save_dir)
    if save_dir.exists() and save_dir.is_dir(): shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load reference images
    ref_paths = [path for path in sorted(ref_dir.glob("*")) if path.suffix.lower() in image_types]
    ref_imgs = []
    for ref_path in ref_paths:
        im = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        ref_imgs.append((ref_path, im))
    print(f"There are {len(ref_imgs)} reference images used\n")    
    ref_h, ref_w = ref_imgs[0][1].shape[:2]

    # calculate averaged reference ROI
    ref_roi = average_ref_roi(ref_imgs, corr_roi)

    # read inputs to derotate
    img_paths = [path for path in sorted(img_dir.glob("*")) if path.suffix.lower() in image_types]

    total = 0
    derotated = 0
    correct = 0
    # loop of images to correlate with the reference 
    for img_path in img_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img.shape[:2] != (ref_h, ref_w):
            print(f"size mismatch: {img_path.name} does not have the same size of ref images")
            continue

        # calculate correlation    
        x1, y1, x2, y2 = corr_roi
        in_roi = img[y1:y2, x1:x2]
        score = corr_green(ref_roi, in_roi)
        # check if needs derotation or not 
        if score >= corr_thresh:
            out_path = save_dir / img_path.name
            cv2.imwrite(str(out_path), img)
            correct += 1
            print(f"Image {img_path.name}: score={score:.4f} -> correct and saved to {out_path.name}")
        else:
            out_path = save_dir / f"{img_path.stem}_corrected{img_path.suffix}"
            corrected = cv2.rotate(img, cv2.ROTATE_180)
            cv2.imwrite(str(out_path), corrected)
            derotated += 1
            print(f"Image {img_path.name}: score={score:.4f} -> derotated and saved to {out_path.name}")
        total+=1    

    print(f"\nTotal images: {total} - correct: {correct} - derotated (rotated 180): {derotated}")

if __name__ == "__main__":
    main()