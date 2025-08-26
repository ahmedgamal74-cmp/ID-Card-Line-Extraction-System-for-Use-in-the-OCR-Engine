# ID Card Line Extraction System (Egyptian IDs) — For OCR Pipelines

Extract clean, per-line (or per-region) crops from scanned/photographed **Egyptian ID** cards—ready for any OCR engine.  
The system includes **classical CV rectification + orientation fixing**, with three interchangeable extraction modes:

- **Classical fixed ROIs** (deterministic layout)
- **YOLO/RT-DETR detection**
- **YOLO segmentation** (used here to produce rectangular crops; masks for previews)

<p align="center">
  <em>Rectify → Derotate (0°/180°) → Extract per-line crops</em>
</p>

---

## Table of Contents

- [Features](#features)  
- [Repo Structure](#repo-structure)  
- [Requirements](#requirements)  
- [Quick Start](#quick-start)  
- [Configuration](#configuration)  
- [End-to-End Pipeline](#end-to-end-pipeline)  
- [Training (Notebooks)](#training-notebooks)  
- [Reproducing Results](#reproducing-results)  
- [Outputs](#outputs)  
- [Evaluation & Metrics](#evaluation--metrics)  
- [Troubleshooting](#troubleshooting)  
- [FAQ](#faq)  
- [Acknowledgements](#acknowledgements)  
- [License](#license)

---

## Features

- **Robust rectification** of ID cards using classical CV (edge detection + Hough + 4-pt transform) to a **standard 1000×631** frame  
- **Orientation normalization**:  
  - 90°/270° handled inside rectification  
  - Final **0°/180°** decision by template-based correlation on a fixed ROI  
- **Three extraction modes**:
  - Classical **fixed ROIs** via `ROIs.json`
  - **Detection** via YOLOv8 or **RT-DETR**
  - **Segmentation** via YOLOv8-SEG (rectangular crops; masks in previews)
- Simple automation via **`main.py` + `config.py`**

---

## Repo Structure

```
ID-Card-Line-Extraction-System-for-Use-in-the-OCR-Engine/
├─ main.py                 # Orchestrates the full pipeline (rectify → derotate → extract)
├─ config.py               # All paths + mode selection ("classic" | "det" | "seg")
│
├─ rectify.py              # Classical rectification → 1000×631 (handles 90°/270°)
├─ corr.py                 # 0° vs 180° derotation using green-channel ROI correlation
│
├─ roi_select.py           # Interactive ROI authoring tool → produces ROIs.json
├─ ROIs.json               # Named fixed boxes (for classical extraction)
├─ roi_extract.py          # Batch crops using ROIs.json (classical mode)
│
├─ infer_det.py            # Detection inference (YOLOv8 or RT-DETR)
├─ infer_seg.py            # Segmentation inference (YOLOv8-SEG; rectangular crops)
│
├─ notebooks/
│  ├─ yolov8s_det.ipynb    # YOLOv8s detection training
│  ├─ rt_detr.ipynb        # RT-DETR detection training
│  └─ yolov8s_seg.ipynb    # YOLOv8s segmentation training
│
├─ models/                 # Put trained weights here (paths configurable)
│  ├─ best_yolov8s_det.pt
│  ├─ best_rtdetr.pt
│  └─ best_yolov8s_seg.pt
│
├─ 0_IDs_input/            # Place raw images here (any orientation)
├─ 0_IDs_derotation_ref/   # A few upright rectified refs for corr.py averaging
├─ 1_IDs_rectified/        # (auto) rectified outputs
├─ 2_IDs_derotated/        # (auto) orientation-correct outputs
├─ 3_ROIs_classic/         # (auto) crops for classical mode
├─ 3_ROIs_det/             # (auto) crops for detection mode
└─ 3_ROIs_seg/             # (auto) crops for segmentation mode
```

---

## Requirements

- **Python** 3.9–3.12  
- **pip** (or conda/mamba)  
- **GPU (CUDA)** optional but recommended for detection/segmentation

Install dependencies:

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install opencv-python numpy imutils scikit-image matplotlib tqdm
pip install ultralytics  # includes YOLOv8 + RT-DETR support via Ultralytics

# If using CUDA, install a torch build compatible with your CUDA version:
# Example (adjust per your CUDA): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU fallback is automatic if CUDA is unavailable.
```

> **Note:** If you hit a torch/ultralytics version mismatch, reinstall torch first (GPU or CPU build), then `pip install ultralytics` again.

---

## Quick Start

1) **Put input images** (any orientation) in `0_IDs_input/`.

2) **Provide derotation references**  
   After you rectify at least a few upright IDs once, copy a handful of their **upright rectified** images into `0_IDs_derotation_ref/`.  
   These are averaged to form a robust correlation template for the final 0°/180° check.

3) **Choose extraction mode** in `config.py`:
   - `"classic"` → use fixed ROIs (`ROIs.json`)
   - `"det"` → detection (YOLO/RT-DETR)
   - `"seg"` → segmentation (YOLOv8-SEG)

4) **Run the full pipeline**:

```bash
python main.py
```

You will see:
- Rectified images → `1_IDs_rectified/`
- Derotated images → `2_IDs_derotated/`
- Crops (per chosen mode) → `3_ROIs_classic/` or `3_ROIs_det/` or `3_ROIs_seg/`

---

## Configuration

Edit **`config.py`** to set all paths and options. Typical fields:

```python
# Input / outputs
rectify_input_dir   = "0_IDs_input"
rectify_output_dir  = "1_IDs_rectified"
derotation_ref_dir  = "0_IDs_derotation_ref"
derotated_dir       = "2_IDs_derotated"

# Extraction mode: "classic" | "det" | "seg"
option              = "classic"

# Classical ROI extraction
rois_json           = "ROIs.json"
classic_out_dir     = "3_ROIs_classic"
crops_output_fromat = "png"          # "png" | "jpg" | "jpeg"

# Detection / Segmentation extraction
det_out_dir         = "3_ROIs_det"
seg_out_dir         = "3_ROIs_seg"
detection_model     = "models/best_rtdetr.pt"    # or models/best_yolov8s_det.pt
segmentation_model  = "models/best_yolov8s_seg.pt"
```

> **OS Note:** Use forward slashes in model paths for cross-platform compatibility.

---

## End-to-End Pipeline

```mermaid
flowchart LR
  A[Raw ID images\n0_IDs_input] --> B[Rectification\nrectify.py → 1000×631]
  B --> C[Derotation (0/180)\ncorr.py (ROI correlation)]
  C --> D{Extraction Mode}

  D -->|classic| E[roi_extract.py\nusing ROIs.json]
  D -->|det| F[infer_det.py\nYOLO/RT-DETR]
  D -->|seg| G[infer_seg.py\nYOLOv8-SEG]

  E --> H[Crops → 3_ROIs_classic/]
  F --> I[Crops + annotated previews → 3_ROIs_det/]
  G --> J[Crops + annotated previews → 3_ROIs_seg/]
```

**Rectification (`rectify.py`)**  
- Preprocess → Canny → Hough Lines → 4-corner intersections → 4-pt transform  
- Standardize to **1000×631** (handles **90°/270°**)  

**Derotation (`corr.py`)**  
- Green-channel correlation vs averaged upright ROI  
- Final 0° vs 180° flip decision  

**Extraction**  
- **Classical**: define ROIs once via `roi_select.py`, then batch crop with `roi_extract.py`  
- **Detection**: YOLOv8/RT-DETR inference (`infer_det.py`)  
- **Segmentation**: YOLOv8-SEG inference (`infer_seg.py`) with rectangular crops

---

## Training (Notebooks)

The `notebooks/` directory contains three training notebooks:

- `yolov8s_det.ipynb` — YOLOv8 **detection**  
- `rt_detr.ipynb` — **RT-DETR** detection  
- `yolov8s_seg.ipynb` — YOLOv8 **segmentation**

**Dataset**  
These notebooks expect a Roboflow dataset in **YOLOv8** format (images + `data.yaml`) aligned with Egyptian IDs.  
Update the download cell or `data_yaml` path if you keep your dataset local.

**Typical training flow (YOLOv8 example):**
```python
from ultralytics import YOLO
model = YOLO("yolov8s.pt")          # or "rtdetr-l.pt", "yolov8s-seg.pt"
model.train(data="path/to/data.yaml",
            imgsz=1024, epochs=50, batch=8, device=0, workers=2, cache=True)
# best weights: runs/detect/train/weights/best.pt (detect)
# or            runs/segment/train/weights/best.pt (segment)
```

Move the resulting `best.pt` into `models/` and set the path in `config.py`.

---

## Reproducing Results

1) **Install environment** (see [Requirements](#requirements)).  
2) **Obtain dataset** and (optionally) **train** models using the notebooks.  
3) **Place weights** under `models/` (or update paths in `config.py`).  
4) **Prepare derotation references** in `0_IDs_derotation_ref/` (a few upright rectified samples).  
5) **Choose mode** in `config.py` (`"classic"`, `"det"`, or `"seg"`).  
6) **Run**:

```bash
python main.py
```

7) **Inspect outputs** in `1_IDs_rectified/`, `2_IDs_derotated/`, and the chosen `3_ROIs_*` folder.  
8) **(Optional) Evaluate** with Ultralytics `model.val()` on your validation/test split.

---

## Outputs

- `1_IDs_rectified/` — Perspective-corrected, standardized images (**1000×631**)  
- `2_IDs_derotated/` — Final upright images (0°) after correlation check  
- `3_ROIs_classic/` — Named crops (e.g., `first_name.png`, `address.png`, `id_number.png`, `code.png`)  
- `3_ROIs_det/` — Detector crops per image + **annotated previews**  
- `3_ROIs_seg/` — Segmentation crops per image + **annotated previews** (masks drawn)

Each crop folder is suitable to feed into your OCR engine.

---

## Evaluation & Metrics

Model-level evaluation (during training) uses Ultralytics’ built-in validation on the dataset `data.yaml`:

- **mAP@0.5**, **mAP@0.5:0.95**  
- **Precision**, **Recall**, **F1**  
- PR curves, confusion matrices, and loss curves are saved under `runs/.../train/`  
- **`best.pt`** is auto-selected by Ultralytics as the checkpoint with the top validation score

At inference time, this repo saves **annotated previews** to visually verify detection/segmentation quality.

---

## Troubleshooting

- **OpenCV “card not found / 4 corners” issues**  
  - Ensure input images have visible card borders (avoid heavy glare/overexposure at edges).  
  - Try cleaner, higher-resolution inputs.

- **Ultralytics / Torch CUDA mismatch**  
  - Install a torch build matching your CUDA version, then reinstall `ultralytics`.  
  - The scripts fall back to CPU if CUDA isn’t available.

- **No crops in classical mode**  
  - Make sure `ROIs.json` exists and its coordinates assume a **1000×631** rectified frame.  
  - Use `roi_select.py` on one rectified sample to author the boxes.

- **Unexpected orientations**  
  - Confirm that rectified images are 1000×631 before derotation.  
  - Ensure `0_IDs_derotation_ref/` contains a few upright rectified references.

---

## FAQ

**Q: Can I use this for other ID templates?**  
Yes. For classical mode, author a new `ROIs.json`. For model-based modes, train on your target template dataset.

**Q: Do I get exact masks for text lines?**  
The segmentation path visualizes masks in annotated previews; crops saved are rectangular. (OCR engines generally accept rect crops.)

**Q: Which mode should I use?**  
- **Classic**: single, stable layout; fastest and deterministic.  
- **Detection/Segmentation**: multiple templates or freer layouts; requires trained weights.

---

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)  
- Roboflow dataset tooling  
- OpenCV, NumPy, scikit-image, imutils

---

## License

Specify your project license here (e.g., MIT, Apache-2.0). If unsure, add a `LICENSE` file to the repo and reference it.

--- 

### Citation

If you use this project in academic work, please cite and/or reference this repository:

```
@software{IDLineExtraction2025,
  author = {Ahmed Gamal Noureddine},
  title  = {ID Card Line Extraction System for Use in the OCR Engine},
  year   = {2025},
  url    = {https://github.com/ahmedgamal74-cmp/ID-Card-Line-Extraction-System-for-Use-in-the-OCR-Engine}
}
```

--- 

**Happy extracting!**
