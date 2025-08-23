#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO label audit & overlay for BraTS slices.

Usage:
  python label_audit.py --data_dir ./data/deb --split train --img_size 640 --samples 32

What it does:
- Scans {split}/labels for YOLO txt files
- Validates ranges (x,y,w,h in [0,1]), non-zero sizes, centers inside image
- Reports class-id distribution (warns if you have IDs > 0 while your model has nc=1)
- Computes anchor coverage per scale (strides 8/16/32) with default YOLO anchors
- Saves overlays (T1ce preferred; fallback flair/t2/t1) with GT boxes drawn
- Prints a short report with suspicious files (zero boxes, tiny boxes, out-of-range)

This does NOT require your training code; it only reads PNGs + TXT labels.
"""
import argparse, sys
from pathlib import Path
import cv2
import numpy as np
from collections import Counter, defaultdict

DEFAULT_ANCHORS = [
    [(10,13),(16,30),(33,23)],
    [(30,61),(62,45),(59,119)],
    [(116,90),(156,198),(373,326)]
]
DEFAULT_STRIDES = [8,16,32]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path containing split/{images,labels}")
    ap.add_argument("--split", default="train", choices=["train","val","test"])
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--samples", type=int, default=24, help="How many overlays to save")
    ap.add_argument("--out_dir", default="label_debug")
    return ap.parse_args()

def read_yolo_txt(path: Path):
    boxes, classes = [], []
    if not path.exists():
        return np.zeros((0,4),np.float32), np.array([],np.int64)
    txt = path.read_text().strip()
    if not txt:
        return np.zeros((0,4),np.float32), np.array([],np.int64)
    for line in txt.splitlines():
        ps = line.strip().split()
        if len(ps) < 5: 
            continue
        try:
            cls = int(float(ps[0]))
            x,y,w,h = map(float, ps[1:5])
            boxes.append([x,y,w,h]); classes.append(cls)
        except Exception:
            continue
    if len(boxes)==0:
        return np.zeros((0,4),np.float32), np.array([],np.int64)
    return np.array(boxes, np.float32), np.array(classes, np.int64)

def choose_image(slice_id: str, images_dir: Path, img_size: int):
    # Prefer T1ce, then flair, t2, t1
    prefs = ["t1ce","flair","t2","t1"]
    for m in prefs:
        for ext in [".png",".PNG"]:
            p = images_dir / f"{slice_id}_{m}{ext}"
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                if img.shape[:2] != (img_size,img_size):
                    img = cv2.resize(img, (img_size,img_size))
                return img, m, p
    return None, None, None

def draw_boxes(img, boxes, color=(0,255,0), thickness=2):
    h,w = img.shape[:2]
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (xc,yc,bw,bh) in boxes:
        x1 = int((xc - bw/2)*w); y1 = int((yc - bh/2)*h)
        x2 = int((xc + bw/2)*w); y2 = int((yc + bh/2)*h)
        cv2.rectangle(out, (max(0,x1),max(0,y1)), (min(w-1,x2),min(h-1,y2)), color, thickness)
    return out

def list_slice_ids(images_dir: Path):
    # Accept patterns: BraTS20_Training_XXXX_slice_YYY_{mod}.png  OR BraTS20_Training_XXXX_YYY_{mod}.png
    ids=set()
    for p in images_dir.glob("*.png"):
        name=p.stem
        if "_slice_" in name:
            parts=name.split("_")
            slice_id="_".join(parts[0:4])  # BraTS20_Training_XXXX_slice_YYY
        else:
            parts=name.split("_")
            slice_id="_".join(parts[0:3])  # BraTS20_Training_XXXX_YYY
        ids.add(slice_id)
    return sorted(ids)

def anchor_coverage(boxes_norm, img_size, anchors_px=DEFAULT_ANCHORS, strides=DEFAULT_STRIDES, anchor_t=4.0):
    # boxes_norm shape (N,4) xywh in [0,1]; convert to grid units per scale then check YOLO matching rule
    if len(boxes_norm)==0:
        return [0,0,0], 0
    wh_px = boxes_norm[:,2:4]*img_size
    matched=0
    matches_per_scale=[0,0,0]
    for si,(anchor_set,stride) in enumerate(zip(anchors_px, strides)):
        anc_grid = np.array(anchor_set, np.float32)/float(stride)  # (3,2) in grid units
        box_grid = wh_px/float(stride)                             # (N,2) in grid
        # ratio in YOLO: max(r,1/r) < anchor_t on both dims -> positive match
        # We'll count if BOTH dims satisfy (approx approaches different impls; this is fine for auditing)
        for b in box_grid:
            r = b[None,:]/anc_grid
            inv = 1.0/np.maximum(r, 1e-6)
            ok = (np.maximum(r, inv) < anchor_t).all(axis=1)  # (3,)
            if ok.any():
                matched+=1
                matches_per_scale[si]+=1
    return matches_per_scale, matched

def main():
    args = parse_args()
    base = Path(args.data_dir) / args.split
    images_dir = base / "images"
    labels_dir = base / "labels"
    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    assert images_dir.exists(), f"{images_dir} not found"
    assert labels_dir.exists(), f"{labels_dir} not found"

    slice_ids = list_slice_ids(images_dir)
    n=len(slice_ids)
    print(f"[INFO] Found {n} slice IDs in {images_dir}")

    bad_files=[]
    zero_box_files=[]
    tiny_box_files=[]
    oob_files=[]

    cls_counter=Counter()
    wh_px_all=[]

    # sample evenly across IDs for overlay
    step = max(1, n//max(1,args.samples))
    picked_ids = slice_ids[::step][:args.samples]

    for sid in slice_ids:
        label_path = labels_dir / f"{sid}.txt"
        boxes, classes = read_yolo_txt(label_path)
        if len(boxes)==0:
            zero_box_files.append(sid)
            continue
        # validate ranges
        x,y,w,h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        if not (np.all((x>=0)&(x<=1)) and np.all((y>=0)&(y<=1)) and np.all((w>=0)&(w<=1)) and np.all((h>=0)&(h<=1))):
            oob_files.append(sid)
        # tiny boxes (area < 3x3 px at img_size)
        ww = (w*args.img_size); hh = (h*args.img_size)
        wh_px_all.extend(np.stack([ww,hh],1).tolist())
        tiny = (ww<3) | (hh<3)
        if np.any(tiny):
            tiny_box_files.append(sid)
        for c in classes:
            cls_counter[c]+=1

        # overlay for picked ones
        if sid in picked_ids:
            img, mod, impath = choose_image(sid, images_dir, args.img_size)
            if img is not None:
                canvas = draw_boxes(img, boxes, (0,255,0), 2)
                cv2.imwrite(str(out_dir / f"{sid}_overlay_{mod}.png"), canvas)

    # coverage
    wh_px_arr = np.array(wh_px_all, np.float32) if len(wh_px_all) else np.zeros((0,2), np.float32)
    cover_per_scale, matched = anchor_coverage(
        np.column_stack([np.zeros((len(wh_px_arr),2)), wh_px_arr/args.img_size]) if len(wh_px_arr) else np.zeros((0,4),np.float32),
        args.img_size
    )

    print("\n===== LABEL AUDIT REPORT =====")
    print(f"Total slice IDs: {n}")
    total_labels = sum(cls_counter.values())
    print(f"Total boxes: {total_labels}")
    print(f"Images with ZERO boxes: {len(zero_box_files)}")
    print(f"Images with out-of-bound coords: {len(oob_files)}")
    print(f"Images with TINY boxes (<3 px side): {len(tiny_box_files)}")
    print(f"Class distribution: {dict(cls_counter)}")
    if len(cls_counter)>0 and (min(cls_counter.keys())<0 or max(cls_counter.keys())>0):
        print("[WARN] You appear to have multiple class IDs. If your model has nc=1, you should remap all to 0.")
    if len(wh_px_arr):
        print(f"Median w,h (px): {np.median(wh_px_arr,0).round(1).tolist()}   Mean: {np.mean(wh_px_arr,0).round(1).tolist()}")
    print(f"Anchor coverage (matches counted) per scale [P3,P4,P5]: {cover_per_scale}  (anchor_t=4.0)")
    print(f"Overlays saved to: {out_dir}")

    # Save suspect lists
    def save_list(lst, name):
        if len(lst)==0: return
        (out_dir/ f"{name}.txt").write_text("\n".join(lst))
    save_list(zero_box_files, "zero_box_files")
    save_list(oob_files, "oob_files")
    save_list(tiny_box_files, "tiny_box_files")

if __name__ == "__main__":
    main()
