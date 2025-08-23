
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BraTS2020 (NIfTI) -> YOLO 2D slice dataset (multimodal)

- Reads cases like BraTS20_Training_XXX with files:
    *_t1.nii.gz, *_t1ce.nii.gz, *_t2.nii.gz, *_flair.nii.gz, *_seg.nii.gz
- Produces YOLO-style structure per split:
    out_root/{split}/images/BraTS20_Training_XXX_slice_YYY_{mod}.png
    out_root/{split}/labels/BraTS20_Training_XXX_slice_YYY.txt  (class x y w h normalized)
- One GT file per slice_id (without modality suffix). Multiple boxes if multiple components.
- Saves a few overlay PNGs for quick verification in: out_root/_debug/{split}/...

Usage examples:
  # If your source root contains subfolders train/, val/, test/ with case dirs inside:
  python brats_to_yolo.py --src_root /path/BraTS2020_TrainingData --out_root ./data/deb --splits train val test

  # If you only want to process a single split folder of cases:
  python brats_to_yolo.py --src_root /path/BraTS2020_TrainingData/train --out_root ./data/deb --splits train

Options:
  --mask_mode WT|TC|ET
    WT = union {1,2,4} (default, matches single-class "tumor")
    TC = union {1,4}    (tumor core)
    ET = {4}            (enhancing tumor only)

  --resize 0
    Leave native size (240x240). Your dataloader resizes later (recommended).
    If >0, images will be resized to NxN BEFORE saving and labels computed accordingly.

  --min_side_px 3  --min_area_px 9
    Remove tiny components (after resize). Set to 0 to keep everything.

Requires: nibabel, numpy, opencv-python, scipy (for binary closing, optional).
"""
import argparse, sys, math, os, random
from pathlib import Path
import numpy as np
import nibabel as nib
import cv2

try:
    from scipy.ndimage import binary_closing
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

MODS = ['t1','t1ce','t2','flair']

def robust_window(vol, p_lo=1, p_hi=99):
    lo = np.percentile(vol, p_lo)
    hi = np.percentile(vol, p_hi)
    if hi <= lo:
        lo, hi = vol.min(), vol.max() if vol.max()>vol.min() else (0,1)
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo + 1e-6)
    return (vol * 255.0).astype(np.uint8)

def choose_mask(seg, mode='WT'):
    seg = np.asarray(seg, dtype=np.int16)
    if mode == 'ET':
        return (seg == 4).astype(np.uint8)
    elif mode == 'TC':
        return ((seg == 1) | (seg == 4)).astype(np.uint8)
    else:  # WT
        return ((seg == 1) | (seg == 2) | (seg == 4)).astype(np.uint8)

def components_to_yolo(mask2d, min_side=3, min_area=9):
    """Return list of [xc, yc, w, h] normalized in [0,1] for each 4-connected component."""
    h, w = mask2d.shape
    if mask2d.max() == 0:
        return []
    # optional morphological closing to tidy jagged edges / tiny holes
    if _HAVE_SCIPY:
        mask2d = binary_closing(mask2d.astype(bool), iterations=1).astype(np.uint8)

    num, cc = cv2.connectedComponents(mask2d, connectivity=4)
    boxes = []
    for label in range(1, num):
        ys, xs = np.where(cc == label)
        if len(xs) == 0:
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        bw = x2 - x1 + 1
        bh = y2 - y1 + 1
        if (min_side and (bw < min_side or bh < min_side)) or (min_area and (bw*bh < min_area)):
            continue
        xc = (x1 + x2 + 1) / 2.0
        yc = (y1 + y2 + 1) / 2.0
        boxes.append([xc / w, yc / h, bw / w, bh / h])
    return boxes

def write_label(path, boxes):
    if not boxes:
        Path(path).write_text("")  # empty file for negatives
        return 0
    with open(path, "w", encoding="utf-8") as f:
        for (xc,yc,bw,bh) in boxes:
            f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    return len(boxes)

def save_overlay(img, boxes, out_path, color=(0,255,0)):
    h,w = img.shape[:2]
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (xc,yc,bw,bh) in boxes:
        x1 = int((xc - bw/2)*w); y1 = int((yc - bh/2)*h)
        x2 = int((xc + bw/2)*w); y2 = int((yc + bh/2)*h)
        cv2.rectangle(canvas, (max(0,x1),max(0,y1)), (min(w-1,x2),min(h-1,y2)), color, 2)
    cv2.imwrite(str(out_path), canvas)

def process_split(src_split_dir: Path, out_root: Path, split: str, mask_mode='WT', resize=0, min_side_px=3, min_area_px=9, debug_samples=48):
    out_images = out_root / split / "images"
    out_labels = out_root / split / "labels"
    out_debug  = out_root / "_debug" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    out_debug.mkdir(parents=True, exist_ok=True)

    # find case folders
    cases = [d for d in src_split_dir.iterdir() if d.is_dir() and d.name.startswith("BraTS20_Training_")]
    cases.sort()
    if len(cases) == 0:
        # maybe src_split_dir is the root that already has cases (no "split" subdir)
        cases = [d for d in src_split_dir.iterdir() if d.is_dir() and d.name.startswith("BraTS20_Training_")]
    assert len(cases) > 0, f"No BraTS20_Training_* cases found in {src_split_dir}"

    saved_debug = 0
    pos_count = 0
    total_slices = 0

    for case_dir in cases:
        case_id = case_dir.name  # e.g., BraTS20_Training_001
        paths = {}
        for m in MODS + ['seg']:
            # e.g., BraTS20_Training_001_flair.nii.gz
            cand = case_dir / f"{case_id}_{m}.nii.gz"
            if not cand.exists():
                raise FileNotFoundError(f"Missing modality {m} for {case_id}: {cand}")
            paths[m] = cand

        # Load volumes (shape ~ 240x240x~155). We assume they are already aligned.
        vols = {m: nib.load(str(paths[m])).get_fdata() for m in MODS}
        seg  = nib.load(str(paths['seg'])).get_fdata()

        # Robust window per volume then to uint8
        vols_u8 = {m: robust_window(vols[m]) for m in MODS}
        H, W, Z = list(vols_u8['t1'].shape)
        assert all(vols_u8[m].shape == (H,W,Z) for m in MODS), "Modalities shape mismatch"

        mask_all = choose_mask(seg, mode=mask_mode).astype(np.uint8)  # (H,W,Z)

        for k in range(Z):  # iterate axial slices
            total_slices += 1
            imgs = {m: vols_u8[m][:,:,k] for m in MODS}      # (H,W)
            mask2d = mask_all[:,:,k]                         # (H,W)

            if resize and resize != H:
                for m in MODS:
                    imgs[m] = cv2.resize(imgs[m], (resize, resize), interpolation=cv2.INTER_LINEAR)
                mask2d = cv2.resize(mask2d, (resize, resize), interpolation=cv2.INTER_NEAREST)
                h = w = resize
            else:
                h, w = imgs['t1'].shape

            boxes = components_to_yolo(mask2d, min_side=min_side_px, min_area=min_area_px)
            if len(boxes) > 0:
                pos_count += 1

            # save images (one PNG per modality)
            slice_id = f"{case_id}_slice_{k:03d}"
            for m in MODS:
                out_img = out_images / f"{slice_id}_{m}.png"
                cv2.imwrite(str(out_img), imgs[m])

            # save labels (one TXT per slice_id, without modality)
            out_lab = out_labels / f"{slice_id}.txt"
            write_label(out_lab, boxes)

            # save a few overlays for sanity check (prefer t1ce)
            if saved_debug < debug_samples and (len(boxes) > 0 or random.random() < 0.05):
                base_mod = 't1ce' if 't1ce' in imgs else 'flair'
                overlay_path = out_debug / f"{slice_id}_overlay_{base_mod}.png"
                save_overlay(imgs[base_mod], boxes, overlay_path)
                saved_debug += 1

    print(f"[{split}] Done. slices: {total_slices}, positive slices: {pos_count}, overlays: {saved_debug}")
    print(f"Images -> {out_images}\nLabels -> {out_labels}\nDebug  -> {out_debug}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help="Root containing split dirs (train/val/test) or a single split with cases")
    ap.add_argument("--out_root", required=True, help="Output YOLO root (will create {split}/images and {split}/labels)")
    ap.add_argument("--splits", nargs="+", default=["train","val","test"], help="Splits to process; if src_root has these as subfolders")
    ap.add_argument("--mask_mode", default="WT", choices=["WT","TC","ET"])
    ap.add_argument("--resize", type=int, default=0, help="Resize NxN before saving (0 = keep native)")
    ap.add_argument("--min_side_px", type=int, default=3)
    ap.add_argument("--min_area_px", type=int, default=9)
    ap.add_argument("--debug_samples", type=int, default=48)
    args = ap.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    # If src_root contains split subfolders, process per split; else treat it as the split itself
    has_subsplits = all((src_root / s).exists() for s in args.splits)
    if has_subsplits:
        for s in args.splits:
            process_split(src_root / s, out_root, s, mask_mode=args.mask_mode,
                          resize=args.resize, min_side_px=args.min_side_px,
                          min_area_px=args.min_area_px, debug_samples=args.debug_samples)
    else:
        # Assume it's a folder of cases and user provided --splits <name>
        if len(args.splits) != 1:
            print("[WARN] src_root has no split subfolders; using the first supplied split name", file=sys.stderr)
        split_name = args.splits[0]
        process_split(src_root, out_root, split_name, mask_mode=args.mask_mode,
                      resize=args.resize, min_side_px=args.min_side_px,
                      min_area_px=args.min_area_px, debug_samples=args.debug_samples)

if __name__ == "__main__":
    main()
