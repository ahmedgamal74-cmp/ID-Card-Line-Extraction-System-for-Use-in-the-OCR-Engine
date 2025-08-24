# python roi_extract.py final/  ROIs.json --outdir ROIs --ext png

import os
import json
import argparse
import cv2
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def list_images(d: Path):
    return [p for p in sorted(d.glob("*")) if p.suffix.lower() in IMG_EXTS]

def process_one_image(image_path, rois, outdir, ext):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return 0, 0
    h, w = img.shape[:2]

    os.makedirs(outdir, exist_ok=True)

    saved = 0
    for roi in rois:
        try:
            name = str(roi["name"]).strip()
            x1, y1, x2, y2 = int(roi["x1"]), int(roi["y1"]), int(roi["x2"]), int(roi["y2"])
        except KeyError as e:
            print(f"Skipping ROI missing key: {e}")
            continue
        except (TypeError, ValueError):
            print(f"Skipping ROI with invalid values: {roi}")
            continue

        # Normalize and clamp to image bounds
        x_lo, x_hi = sorted((x1, x2))
        y_lo, y_hi = sorted((y1, y2))
        x_lo = clamp(x_lo, 0, w)
        x_hi = clamp(x_hi, 0, w)
        y_lo = clamp(y_lo, 0, h)
        y_hi = clamp(y_hi, 0, h)

        if x_hi <= x_lo or y_hi <= y_lo:
            print(f"Skipping ROI '{name}': empty/invalid after clamping.")
            continue

        crop = img[y_lo:y_hi, x_lo:x_hi]

        # Safe filename
        safe_name = "".join(c if c.isalnum() or c in ("-","_") else "_" for c in name)
        out_path = os.path.join(outdir, f"{safe_name}.{ext}")

        if ext.lower() in ("jpg","jpeg"):
            ok = cv2.imwrite(out_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            ok = cv2.imwrite(out_path, crop)

        if ok:
            saved += 1
            print(f"[{Path(image_path).name}] Saved ROI '{name}' -> {out_path}")
        else:
            print(f"[{Path(image_path).name}] Failed to save ROI '{name}'.")

    return saved, len(rois)

def main():
    p = argparse.ArgumentParser(description="Crop ROIs from an ID card image using a JSON spec.")
    p.add_argument("image_path", help="Path to a rectified ID image OR a folder of images")
    p.add_argument("rois_json", help="Path to the ROI JSON file (list of ROI objects)")
    p.add_argument("--outdir", default="roi_crops", help="Output directory (default: roi_crops)")
    p.add_argument("--ext", default="png", choices=["png","jpg","jpeg"], help="Output image format (default: png)")
    args = p.parse_args()

    in_path = Path(args.image_path)

    # Load ROIs (unchanged logic)
    with open(args.rois_json, "r", encoding="utf-8") as f:
        rois = json.load(f)
    if not isinstance(rois, list):
        raise SystemExit("ROI JSON must be a list of ROI objects.")

    total_saved = 0
    total_expected = 0

    if in_path.is_dir():
        # Folder mode: process each image; save into outdir/<image_stem>/
        images = list_images(in_path)
        if not images:
            raise SystemExit(f"No images found in folder: {in_path}")
        for img_p in images:
            sub_out = Path(args.outdir) / img_p.stem
            saved, expected = process_one_image(str(img_p), rois, str(sub_out), args.ext)
            total_saved += saved
            total_expected += expected
        print(f"Done. Saved {total_saved}/{total_expected * len(images) // max(len(images),1)} crops across {len(images)} images to '{args.outdir}'.")
    else:
        # Single image mode: exact original behavior
        os.makedirs(args.outdir, exist_ok=True)
        saved, expected = process_one_image(str(in_path), rois, args.outdir, args.ext)
        print(f"Done. Saved {saved}/{expected} crops to '{args.outdir}'.")

if __name__ == "__main__":
    main()