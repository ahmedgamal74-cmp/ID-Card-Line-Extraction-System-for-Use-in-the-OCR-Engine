# simple main file to run the piepline
import sys, subprocess
from pathlib import Path
from config import *

device=device
print("Using:", device)

# rectify
print("Rectification started...")
Path(rectify_output_dir).mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, "rectify.py", rectify_input_dir, rectify_output_dir], check=True)

# derotate
print("\nDerotatation started...")
Path(derotated_dir).mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, "corr.py", "--ref_dir", derotation_ref_dir, 
                                "--img_dir", rectify_output_dir, "--save_dir", derotated_dir], check=True)

# extraction mode (classic/detection/segmentation)
if option == "det":
    print(f"\nCrops (ROIs) extraction using {detection_model} detection model started...")
    Path(det_out_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "infer_det.py", "--input_dir", derotated_dir, "--save_dir", det_out_dir], check=True)
elif option == "seg":
    print(f"\nCrops (ROIs) extraction using yolo segmentation model started...")
    Path(seg_out_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "infer_seg.py", "--input_dir", derotated_dir, "--save_dir", seg_out_dir], check=True)
else:
    print(f"\nCrops (ROIs) extraction using classical method started...")
    Path(classic_out_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "roi_extract.py", derotated_dir, rois_json, 
                                    "--outdir", classic_out_dir, "--ext", crops_output_fromat], check=True)

