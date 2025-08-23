#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrected YOLO label audit for BraTS-style slice naming.

Fixes a previous bug: when filenames contain "_slice_YYY_mod.png", we must include
the numeric YYY in the slice_id. This script uses regex to extract slice_id + modality.

Expected naming (any of these):
  BraTS20_Training_001_slice_089_t1ce.png   -> slice_id = BraTS20_Training_001_slice_089
  BraTS20_Training_001_089_t1ce.png         -> slice_id = BraTS20_Training_001_089

Label filename must be: labels/{slice_id}.txt   (no modality suffix).
"""
import argparse, re, sys
from pathlib import Path
import cv2, numpy as np
from collections import Counter

PATTERNS = [
    re.compile(r'^(BraTS20_Training_\d+_slice_\d{3})_(t1ce|t1|t2|flair)\.png$', re.IGNORECASE),
    re.compile(r'^(BraTS20_Training_\d+_\d{3})_(t1ce|t1|t2|flair)\.png$', re.IGNORECASE),
]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Root containing {split}/images and {split}/labels")
    ap.add_argument("--split", default="train", choices=["train","val","test"])
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--out_dir", default="label_debug")
    return ap.parse_args()

def parse_name(name: str):
    for pat in PATTERNS:
        m = pat.match(name)
        if m:
            return m.group(1), m.group(2)
    return None, None

def read_yolo_txt(path: Path):
    if not path.exists():
        return None, []  # None -> missing
    txt = path.read_text().strip()
    if not txt:
        return True, []  # exists but empty
    boxes=[]
    for line in txt.splitlines():
        ps = line.strip().split()
        if len(ps) >= 5:
            x,y,w,h = map(float, ps[1:5])
            boxes.append([x,y,w,h])
    return True, boxes

def draw_overlay(img, boxes, out_path):
    h,w = img.shape[:2]
    can = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (xc,yc,bw,bh) in boxes:
        x1 = int((xc-bw/2)*w); y1 = int((yc-bh/2)*h)
        x2 = int((xc+bw/2)*w); y2 = int((yc+bh/2)*h)
        cv2.rectangle(can, (max(0,x1),max(0,y1)), (min(w-1,x2),min(h-1,y2)), (0,255,0), 2)
    cv2.imwrite(str(out_path), can)

def main():
    args = parse_args()
    base = Path(args.data_dir)/args.split
    images_dir = base/"images"
    labels_dir = base/"labels"
    out_dir = Path(args.out_dir)/args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    assert images_dir.exists(), images_dir
    assert labels_dir.exists(), labels_dir

    slice_to_mods = {}
    for p in images_dir.glob("*.png"):
        sid, mod = parse_name(p.name)
        if sid is None: 
            continue
        slice_to_mods.setdefault(sid, set()).add(mod)

    print(f"[INFO] Found {len(slice_to_mods)} slice IDs in {images_dir}")

    missing=0; empty=0; with_boxes=0; total_boxes=0
    missing_examples=[]; empty_examples=[]

    # save overlays for a subset
    saved=0
    for i,(sid,mods) in enumerate(sorted(slice_to_mods.items())):
        lab = labels_dir / f"{sid}.txt"
        exists, boxes = read_yolo_txt(lab)
        if exists is None:
            missing += 1
            if len(missing_examples) < 8: missing_examples.append(sid)
            continue
        if len(boxes)==0:
            empty += 1
            if len(empty_examples) < 8: empty_examples.append(sid)
        else:
            with_boxes += 1
            total_boxes += len(boxes)

        # save some overlays (prefer t1ce)
        if saved < args.samples:
            # pick an available modality
            mod = "t1ce" if "t1ce" in mods else sorted(list(mods))[0]
            img_path = images_dir / f"{sid}_{mod}.png"
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                draw_overlay(img, boxes, out_dir / f"{sid}_overlay_{mod}.png")
                saved += 1

    print("\n===== LABEL AUDIT REPORT (fixed) =====")
    print(f"Total slice IDs: {len(slice_to_mods)}")
    print(f"Labels MISSING: {missing}")
    print(f"Labels EMPTY  : {empty}")
    print(f"Slices WITH boxes: {with_boxes}   (total boxes: {total_boxes})")
    if missing_examples:
        print("Missing examples (first few):")
        for s in missing_examples: print("  ", s)
    if empty_examples:
        print("Empty examples (first few):")
        for s in empty_examples: print("  ", s)
    print(f"Overlays saved to: {out_dir}")

if __name__ == "__main__":
    main()
