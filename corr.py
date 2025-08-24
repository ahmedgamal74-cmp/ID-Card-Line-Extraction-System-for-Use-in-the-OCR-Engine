# Usage:
#   python corr.py --ref_dir 0_IDs_derotation_ref/ --img_dir 1_IDs_rectified/  --save_dir 2_IDs_derotated/
#
# Notes:
# - Uses SAME-AREA correlation on a fixed ROI (no mirrored check).
# - If correlation < THRESH, the image is assumed ROTATED 180° and is auto-rotated
#   then saved as "<name>_corrected.<ext>" next to the original.
# - All images (refs & inputs) must share the SAME dimensions.
# - Tweak ROI and THRESH constants below if needed.

import argparse
from pathlib import Path
import cv2
import numpy as np

# ======= EDIT THESE IF NEEDED =======
# Fixed ROI in pixel coords (x1, y1, x2, y2)
ROI = (416, 3, 908, 156)
# Decision threshold for "same-area" correlation (Pearson/ZNCC on green channel)
THRESH = 0.10
# ====================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_images(d: Path):
    return [p for p in sorted(d.glob("*")) if p.suffix.lower() in IMG_EXTS]

def clamp_roi(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Empty ROI after clamping: {(x1,y1,x2,y2)} for size {w}x{h}")
    return x1, y1, x2, y2

def pearson_corr(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    return float((a * b).mean() / (a.std() * b.std() + 1e-6))

def corr_green(roi_ref_bgr, roi_in_bgr):
    # Zero-mean normalized cross-correlation on the green channel (lighting-robust)
    return pearson_corr(roi_ref_bgr[..., 1], roi_in_bgr[..., 1])

def average_ref_roi(ref_imgs, roi):
    """Average the reference ROI pixel-wise to get a stable template."""
    acc = None
    count = 0
    x1 = y1 = x2 = y2 = None
    for _, r in ref_imgs:
        h, w = r.shape[:2]
        x1, y1, x2, y2 = clamp_roi(*roi, w, h)
        roi_r = r[y1:y2, x1:x2].astype(np.float32)
        if acc is None:
            acc = np.zeros_like(roi_r, dtype=np.float32)
        acc += roi_r
        count += 1
    mean_roi = (acc / max(count, 1)).astype(np.uint8)
    return mean_roi, (x1, y1, x2, y2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", required=True, help="folder with upright reference images")
    ap.add_argument("--img_dir", required=True, help="folder with input images to check")
    ap.add_argument("--save_dir", required=True, help="folder to save upright versions")
    args = ap.parse_args()

    ref_dir = Path(args.ref_dir)
    img_dir = Path(args.img_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load references
    ref_paths = list_images(ref_dir)
    if not ref_paths:
        raise SystemExit(f"No reference images found in: {ref_dir}")
    ref_imgs = []
    for p in ref_paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            print(f"Skip unreadable ref: {p}")
            continue
        ref_imgs.append((p, im))
    if not ref_imgs:
        raise SystemExit("No readable reference images.")
    ref_h, ref_w = ref_imgs[0][1].shape[:2]

    # Build averaged reference ROI
    ref_roi, roi_used = average_ref_roi(ref_imgs, ROI)
    print(f"ROI used: {roi_used} (width {roi_used[2]-roi_used[0]}, height {roi_used[3]-roi_used[1]})")
    print(f"Threshold: {THRESH:.2f}")

    # Process inputs
    img_paths = list_images(img_dir)
    if not img_paths:
        raise SystemExit(f"No input images found in: {img_dir}")

    saved_ok = 0
    fixed = 0
    skipped = 0

    for p in img_paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            print(f"Skip unreadable: {p.name}")
            continue
        if im.shape[:2] != (ref_h, ref_w):
            print(f"Skip size mismatch: {p.name} has {im.shape[:2]}, expected {(ref_h, ref_w)}")
            continue

        x1, y1, x2, y2 = roi_used
        in_roi = im[y1:y2, x1:x2]
        score = corr_green(ref_roi, in_roi)
        is_upright = score >= THRESH

        if is_upright:
            out_path = save_dir / p.name
            cv2.imwrite(str(out_path), im)
            saved_ok += 1
            print(f"{p.name}: score={score:.4f} -> NOT_ROTATED → saved {out_path.name}")
        else:
            out_path = save_dir / f"{p.stem}_corrected{p.suffix}"
            corrected = cv2.rotate(im, cv2.ROTATE_180)
            cv2.imwrite(str(out_path), corrected)
            fixed += 1
            print(f"{p.name}: score={score:.4f} -> ROTATED → saved {out_path.name}")

    print(f"\nDone. Saved OK (original): {saved_ok} | Saved corrected (rotated 180°): {fixed} | Skipped: {skipped}")

if __name__ == "__main__":
    main()
