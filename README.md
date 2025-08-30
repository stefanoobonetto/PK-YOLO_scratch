# Multimodal PK‑YOLO for Brain Tumor Detection 

## Table of Contents
1. [Overview](#1-overview)
2. [Pipeline at a Glance](#2-pipeline-at-a-glance)
3. [System Components](#3-system-components)
   - [3.1 Data Layer (`brats_dataset.py`)](#31-data-layer-brats_datasetpy)
   - [3.2 Model (`multimodal_pk_yolo.py`)](#32-model-multimodal_pk_yolopy)
   - [3.3 Loss (`loss.py`)](#33-loss-complete_yolo_losspy)
   - [3.4 Training (`training_script.py`)](#34-training-training_scriptpy)
   - [3.5 In‑training Visualization (`train_visualizer.py`)](#35-in-training-visualization-train_visualizerpy)
   - [3.6 Early Stopping (`early_stopping.py`)](#36-early-stopping-early_stoppingpy)
   - [3.7 Inference (`inference.py`, `utils_inference.py`)](#37-inference-inferencepy-utils_inferencepy)
   - [3.8 Configuration & Helpers (`config.py`, `utils.py`)](#38-configuration--helpers-configpy-utilspy)
4. [Repository Layout](#4-repository-layout)
5. [Environment](#5-environment)
6. [Data Preparation](#6-data-preparation)
7. [How the Pipeline Works (Step‑by‑Step)](#7-how-the-pipeline-works-step-by-step)
8. [Training: Commands](#8-training-commands)
9. [Inference: Commands](#9-inference-commands)
10. [Outputs & Logging](#10-outputs--logging)
11. [Design Choices](#11-design-choices)
12. [Known Quirks & Fixes](#12-known-quirks--fixes)
13. [FAQ / Troubleshooting](#13-faq--troubleshooting)
14. [License & Citation](#14-license--citation)

---

## 1) Overview

This project adapts the YOLO detection paradigm to **multimodal MRI slices** (BraTS: `t1`, `t1ce`, `t2`, `flair`). It detects tumors via bounding boxes on 2D slices using:
- a lightweight backbone with **Depth‑wise separable blocks + SE (Squeeze‑and‑Excitation)**,
- **cross‑modal channel attention** to fuse the 4 modalities,
- an FPN neck and **YOLO‑style multi‑scale detection head**,
- a custom **YOLOLoss** (CIoU for box, BCE for objectness/class).

**Labels:** YOLO format `[class cx cy w h]` normalized to `[0,1]`.

---

## 2) Pipeline at a Glance

```mermaid
flowchart LR
    A[4‑channel slice
(t1,t1ce,t2,flair)] --> B[Albumentations
+ Normalize]
    B --> C[DataLoader
(collate bboxes)]
    C --> D[Backbone
DWConv + SE]
    D --> E[FPN (P3,P4,P5)]
    E --> F[YOLO Head
(anchors)]
    F --> G[Loss: CIoU + BCE]
    G --> H[Optimizer + Scheduler
(AMP optional)]
    H --> I[Checkpoints + Vis]
    I --> J[Inference
Decode + NMS]
    J --> K[JSONL + Overlays]
```

---

## 3) System Components

### 3.1 Data Layer (`brats_dataset.py`)
- **Input**: four `.png` modalities per slice (`t1`, `t1ce`, `t2`, `flair`) stacked to shape `(H,W,4)`.
- **Normalization** (per channel):
  - `mean=[0.485, 0.456, 0.406, 0.485]`
  - `std=[0.229, 0.224, 0.225, 0.229]`
- **Augmentations** (train): resize → flip → brightness/contrast → gamma → noise → normalize.
- **Labels**: YOLO txt with `class cx cy w h` normalized; one line per tumor.
- **Collate**: pads variable number of boxes to a fixed tensor; returns `images`, `bboxes`, `labels`, `slice_ids`.

**Expected layout**
```
data/
├─ train/
│  ├─ images/  # BraTS20_..._slice_###_{t1|t1ce|t2|flair}.png
│  └─ labels/  # BraTS20_..._slice_###.txt
├─ val/
│  ├─ images/
│  └─ labels/
└─ test/
   ├─ images/
   └─ labels/   # optional
```

---

### 3.2 Model (`multimodal_pk_yolo.py`)
- **Backbone**: DWConv blocks + **SE**, with residuals; **ChannelAttention** emphasizes informative modality channels.
- **Neck**: FPN with lateral and smoothing convolutions (multi‑scale features).
- **Head**: Shared convs → per‑scale heads for:
  - **cls**: `num_anchors * num_classes`
  - **box**: `num_anchors * 4`
  - **obj**: `num_anchors`
- **Anchors** (typical YOLO triplets across P3/P4/P5) tuned for small–large tumors.

Factory:
```python
from multimodal_pk_yolo import create_model
model = create_model(num_classes=1, input_channels=4, pretrained_path=None, device='cuda')
```

---

### 3.3 Loss (`complete_yolo_loss.py`)
- **Assignments**: anchor/grid aware target building with neighbor offsets.
- **Box term**: CIoU on decoded predictions; **Obj**/**Cls** via BCEWithLogits.
- **Autobalance**: optional per‑level scaling; numerical guards (clamps/eps).

---

### 3.4 Training (`training_script.py`)
- Orchestrates **model + loss + optimizer (AdamW/SGD) + scheduler (Cosine/Step/Plateau) + AMP + EarlyStopping**.
- Builds `DataLoader`s, logs per‑batch losses, saves **checkpoints** and **visualizations**.

---

### 3.5 In‑training Visualization (`train_visualizer.py`)
- Decodes head outputs during training to create **overlay images** (GT vs Pred) saved periodically.

---

### 3.6 Early Stopping (`early_stopping.py`)
- Stops when validation loss stalls by `min_delta` for `patience` epochs.

---

### 3.7 Inference (`inference.py`, `utils_inference.py`)
- CLI parses dataset/weights/thresholds.
- Decoding: grid + anchor transforms → NMS → normalized `[cx,cy,w,h]`.
- Outputs **JSONL** per slice and optional overlays to disk.

---

### 3.8 Configuration & Helpers (`config.py`, `utils.py`)
- `config.py`: default config + a `SimpleConfig` wrapper; supports CLI overrides (see §12 for a small fix).
- `utils.py`: `get_train_arg_parser()` (training CLI), label/util helpers.

---

## 4) Repository Layout

> The code imports expect a `utils/` package. Create it (or update imports accordingly).

```
project/
├─ brats_dataset.py
├─ complete_yolo_loss.py
├─ inference.py
├─ multimodal_pk_yolo.py
├─ training_script.py
├─ utils/                           # ← create this folder
│  ├─ __init__.py                   # (empty file is fine)
│  ├─ config.py
│  ├─ early_stopping.py
│  ├─ train_visualizer.py
│  ├─ utils.py
│  └─ utils_inference.py
└─ README.md
```

If you prefer a flat layout, change imports like `from utils.config import ...` → `from config import ...` etc. in `training_script.py` and `inference.py`.

---

## 5) Environment

```bash
# Choose the right torch build for your CUDA/CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core libs
pip install albumentations opencv-python numpy matplotlib tqdm
```

> **Tip**: On Windows, PyTorch DataLoader workers > 0 can be flaky; this repo already forces `num_workers=0` on Windows.

---

## 6) Data Preparation

Place your BraTS‑derived PNG slices + YOLO labels as shown in §3.1.  
**Label line format** (normalized):
```
0 0.512 0.478 0.180 0.140   # class cx cy w h
```

---

## 7) How the Pipeline Works (Step‑by‑Step)

1. **Index slices**: the dataset enumerates slice IDs and finds the 4 modality files.
2. **Load & stack**: `(H,W,4)` array built as `[t1, t1ce, t2, flair]`.
3. **Augment & normalize**: Albumentations for train; validation uses deterministic resize+normalize.
4. **Batch & collate**: variable‑len bboxes padded to `(B, Nmax, 4)` with `labels` `(B, Nmax)`.
5. **Forward pass**:
   - Backbone (DWConv+SE) extracts features; ChannelAttention emphasizes modalities.
   - FPN yields multi‑scale maps (P3/P4/P5).
   - YOLO head predicts `cls`, `box`, `obj` per anchor/grid cell.
6. **Loss**:
   - Targets built per scale/anchor.
   - CIoU on decoded boxes; BCE on objectness & class.
7. **Optimize**:
   - AdamW (default) + CosineAnnealingLR; optional mixed precision (AMP).
   - EarlyStopping on validation loss.
8. **Artifacts**:
   - `outputs/checkpoints/{best_model.pth, final_model.pth}`
   - `outputs/training_visualizations/*.png`
9. **Inference**:
   - Decode (sigmoid + grid + anchors) → NMS → save JSONL and overlays.

---

## 8) Training: Commands

> **Important:** The current `training_script.py` parses CLI args but doesn’t apply them to the config (see §12 for a 2‑line fix). Until you patch it, keep your data at `./data` so defaults work.

**Minimal run (after creating `utils/` package)**:
```bash
python training_script.py --data_dir ./data
```

**Recommended (after patch in §12 to apply CLI → config):**
```bash
python training_script.py   --data_dir /path/to/data   --output_dir outputs   --batch_size 8   --epochs 100   --lr 1e-3   --img_size 640   --workers 4   --mixed_precision
```

**What you’ll see**: per‑batch loss breakdown (Box/Obj/Cls), LR, epoch summaries, and periodic visualizations saved to disk.

---

## 9) Inference: Commands

```bash
python inference.py   --data_dir /path/to/data   --split val   --weights outputs/checkpoints/best_model.pth   --img_size 640   --batch_size 8   --num_workers 2   --conf_thresh 0.25   --iou_thresh 0.45   --max_dets 50   --save_dir runs/inference/exp   --save_vis   --json_name predictions.jsonl
```

**Outputs**
- JSONL records like:
  ```json
  {"slice_id": "BraTS20_Training_0001_slice_000", "detections": [
    {"bbox":[0.49,0.52,0.18,0.14], "confidence":0.93, "class":0}
  ]}
  ```
- Overlays (if `--save_vis`) at `runs/inference/exp/vis/*.png`.

> Use `--half` for FP16 inference on CUDA.

---

## 10) Outputs & Logging

- `outputs/checkpoints/` — periodic + best/final checkpoints
- `outputs/training_visualizations/` — GT/Pred overlays during training
- `runs/inference/exp/` — JSONL predictions + optional overlays
- Console logs — losses, LR, epoch timing; errors include CUDA OOM handling

---

## 11) Design Choices

- **Multimodal fusion**: 4‑channel input + explicit channel attention to leverage modality complementarity.
- **Anchors**: three per scale (P3/P4/P5) to cover small–large tumor extents.
- **Loss**: CIoU stabilizes localization; BCE for obj/cls is standard for one‑class detection.
- **Training stability**: AMP, early stopping, and guarded numerics in loss/decoding.

---

## 12) Known Quirks & Fixes

### (A) Package layout
`training_script.py` and `inference.py` import from `utils.*`. Create the package:
```bash
mkdir -p utils
touch utils/__init__.py
# move files into the package
git mv config.py early_stopping.py train_visualizer.py utils.py utils_inference.py utils/ 2>/dev/null || mv config.py early_stopping.py train_visualizer.py utils.py utils_inference.py utils/
```

### (B) CLI args not applied to config
`training_script.py` currently ignores parsed CLI flags. Patch `main()` as below so CLI overrides take effect:

```python
# training_script.py
from utils.config import create_default_config, SimpleConfig  # add create_default_config import

def main():
    parser = get_train_arg_parser()
    args = parser.parse_args()

    # BEFORE: config = SimpleConfig()
    # AFTER:
    cfg_dict = create_default_config(args)   # apply CLI → config dict
    config = SimpleConfig(cfg_dict)

    ...
```
After this change, flags like `--data_dir`, `--batch_size`, etc. will drive training.

---

## 13) FAQ / Troubleshooting

- **`ModuleNotFoundError: utils...`**  
  Create the `utils/` package and move helper files as in §12(A), or change imports to match a flat layout.

- **Training crashes without `--data_dir`.**  
  The parser requires it. Until §12(B) is applied, it won’t change the config, so also keep your data at `./data` or patch as shown.

- **Windows dataloader deadlocks.**  
  The script already forces `num_workers=0` on Windows. You can increase on Linux.

- **OOM on GPU.**  
  Lower `--batch_size` and/or `--img_size`. AMP is available via `--mixed_precision`.

---

## 14) License & Citation

Add your preferred license (e.g., MIT) and cite if used in academic work:

```
@misc{yourname2025multimodalpkyolo,
  title  = {Multimodal PK-YOLO for Brain Tumor Detection},
  author = {Your Name},
  year   = {2025},
  note   = {Final Project},
}
```

---

### TL;DR (Quick Start)

```bash
# 1) Env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install albumentations opencv-python numpy matplotlib tqdm

# 2) Repo layout
mkdir -p utils && touch utils/__init__.py
mv config.py early_stopping.py train_visualizer.py utils.py utils_inference.py utils/

# 3) Data under ./data (see structure in §3.1)

# 4) (Recommended) Patch training_script.py as in §12(B)

# 5) Train
python training_script.py --data_dir ./data --batch_size 8 --epochs 100 --lr 1e-3 --img_size 640 --mixed_precision

# 6) Inference
python inference.py --data_dir ./data --split val --weights outputs/checkpoints/best_model.pth --save_vis
```
