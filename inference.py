import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import csv
import json
import math
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
torch.set_num_threads(1)
from torch.utils.data import DataLoader

from multimodal_pk_yolo import create_model
from brats_dataset import BraTSDataset, collate_fn
from utils.utils import get_inference_arg_parser

from utils.utils_inference import (
    decode_yolo_predictions, save_visualization, time_synchronized
)

# ------------------ Geometry helpers ------------------
def cxcywh_to_xyxy_norm(b):
    cx, cy, w, h = b
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return [x1, y1, x2, y2]

def box_iou_xyxy_norm(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-12
    return inter / union

def greedy_match_dets_to_gt(pred_xyxy, pred_cls, pred_scores, gt_xyxy, gt_cls, iou_thr=0.50):
    """One-to-one greedy matching by IoU, class-aware."""
    n_pred = len(pred_xyxy)
    n_gt = len(gt_xyxy)
    used = [False] * n_gt
    tp = [0] * n_pred
    fp = [0] * n_pred
    ious = [0.0] * n_pred

    for i in range(n_pred):
        best_j = -1
        best_iou = 0.0
        for j in range(n_gt):
            if used[j]:
                continue
            if int(pred_cls[i]) != int(gt_cls[j]):
                continue
            iou = box_iou_xyxy_norm(pred_xyxy[i], gt_xyxy[j])
            if iou >= iou_thr and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            used[best_j] = True
            tp[i] = 1
            ious[i] = best_iou
        else:
            fp[i] = 1
    fn = used.count(False)
    return tp, fp, fn, ious

def _ap_from_pr(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for k in range(mpre.size - 1, 0, -1):
        mpre[k - 1] = max(mpre[k - 1], mpre[k])
    rs = np.linspace(0, 1, 101)
    p = np.array([mpre[mrec >= r].max() if np.any(mrec >= r) else 0.0 for r in rs])
    return p.mean()

def _compute_ap_for_class_at_iou(preds_cls, gts_cls, iou=0.50):
    if len(preds_cls) == 0:
        return float("nan")
    npos = sum(len(v) for v in gts_cls.values())
    if npos == 0:
        return float("nan")
    
    preds_cls = sorted(preds_cls, key=lambda x: x[1], reverse=True)
    tp = np.zeros(len(preds_cls), dtype=np.float32)
    fp = np.zeros(len(preds_cls), dtype=np.float32)
    matched = {img_id: [False]*len(boxes) for img_id, boxes in gts_cls.items()}

    for i, (img_id, score, box) in enumerate(preds_cls):
        gts = gts_cls.get(img_id, [])
        if not gts:
            fp[i] = 1.0
            continue
        ious = np.array([box_iou_xyxy_norm(box, gt) for gt in gts], dtype=np.float32)
        j = int(ious.argmax()) if ious.size else -1
        if ious.size and ious[j] >= iou and not matched[img_id][j]:
            matched[img_id][j] = True
            tp[i] = 1.0
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / (npos + 1e-12)
    prec = tp_cum / (tp_cum + fp_cum + 1e-12)
    return _ap_from_pr(rec, prec)

def compute_map(all_preds_raw, all_gts, num_classes, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = [0.50 + 0.05*k for k in range(10)]

    gts_per_cls = [defaultdict(list) for _ in range(num_classes)]
    preds_per_cls = [[] for _ in range(num_classes)]
    for img_id, gts in all_gts.items():
        for cls_id, box in gts:
            gts_per_cls[int(cls_id)][img_id].append(box)
    for img_id, preds in all_preds_raw.items():
        for cls_id, score, box in preds:
            preds_per_cls[int(cls_id)].append((img_id, float(score), box))

    ap_at_iou = []
    for t in iou_thresholds:
        ap_c = []
        for c in range(num_classes):
            ap = _compute_ap_for_class_at_iou(preds_per_cls[c], gts_per_cls[c], iou=t)
            if not math.isnan(ap):
                ap_c.append(ap)
        ap_at_iou.append(float(np.mean(ap_c)) if len(ap_c) else float("nan"))
    
    res = {
        "mAP50": ap_at_iou[0] if len(ap_at_iou) else float("nan"),
        "mAP50_95": float(np.nanmean(ap_at_iou)) if len(ap_at_iou) else float("nan"),
        "AP_per_IoU": {f"{t:.2f}": ap for t, ap in zip(iou_thresholds, ap_at_iou)}
    }
    return res

def compute_map_with_threshold(all_preds_raw, all_gts, num_classes, thr, iou_thresholds=None):
    filtered = {k: [(c,s,b) for (c,s,b) in v if s >= thr] for k, v in all_preds_raw.items()}
    return compute_map(filtered, all_gts, num_classes, iou_thresholds=iou_thresholds)

# ------------------ Arg parsing ------------------
def parse_sweep(s):
    if s is None or s == "":
        return []
    s = s.strip()
    vals = []
    if ":" in s:
        a, b, step = s.split(":")
        a = float(a); b = float(b); step = float(step)
        x = a
        while x <= b + 1e-9:
            vals.append(round(x, 6))
            x += step
    else:
        for tok in s.split(","):
            tok = tok.strip()
            if tok:
                vals.append(float(tok))
    vals = sorted(set(vals))
    return vals

def load_model_with_spark_support(weights_path, num_classes, input_channels, device, half=False):
    """
    Load model with automatic SparK detection from checkpoint.
    """
    # First, try to load checkpoint to detect SparK usage
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # Check if this model uses SparK pretraining
    use_spark_pretrained = False
    spark_pretrained_path = None
    
    if 'use_spark_pretrained' in checkpoint:
        use_spark_pretrained = checkpoint['use_spark_pretrained']
        spark_pretrained_path = checkpoint.get('spark_pretrained_path', None)
        logging.info(f"Detected SparK pretrained model: {use_spark_pretrained}")
        if spark_pretrained_path:
            logging.info(f"SparK weights path: {spark_pretrained_path}")
    
    # Create model with appropriate configuration
    model = create_model(
        num_classes=num_classes,
        input_channels=input_channels,
        pretrained_path=None,  # We'll load the state dict manually
        device=device,
        use_spark_pretrained=use_spark_pretrained,
        spark_pretrained_path=spark_pretrained_path  # This will be used for backbone initialization
    )
    
    # Load the full model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            logging.info(f"Loaded model from epoch {checkpoint['epoch']}")
        if 'best_loss' in checkpoint:
            logging.info(f"Best training loss: {checkpoint['best_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    if half:
        model.half()
    
    return model

# ------------------ Main ------------------
def inference():
    args = get_inference_arg_parser().parse_args()
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("inference")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    half = args.half and device.type == "cuda"

    # Load model with SparK support
    model = load_model_with_spark_support(
        args.weights, args.num_classes, args.input_channels, device.type, half=half
    ).to(device)

    # Data
    ds = BraTSDataset(args.data_dir, split=args.split, img_size=args.img_size, augment=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=(device.type=="cuda"), collate_fn=collate_fn)

    # Thresholds to evaluate
    thr_list = parse_sweep(args.conf_sweep)
    if not thr_list:
        thr_list = [args.conf_thresh]
    min_thr_for_decode = min(0.001, min(thr_list) - 1e-6) if thr_list else 0.001

    # Output files
    pred_path = save_dir/"predictions.jsonl"
    metrics_path = save_dir/"metrics.csv"
    pr_tuning_path = save_dir/"pr_tuning.csv"
    best_thr_path = save_dir/"best_threshold.json"
    if pred_path.exists():
        pred_path.unlink()

    # Aggregators
    all_preds_raw = defaultdict(list)
    all_gts = defaultdict(list)
    n_images = 0
    n_pos_images = 0
    t0 = time_synchronized()

    # Pass 1: run model, decode once at low conf, aggregate preds & gts
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            images = batch["images"].to(device, non_blocking=True)
            if half:
                images = images.half()
            if images.dtype != torch.float16 and images.dtype != torch.float32:
                images = images.float()

            # forward
            t1 = time_synchronized()
            preds = model(images)
            infer_time = time_synchronized() - t1

            # anchors from model
            anchors = model.anchors if hasattr(model, "anchors") else None
            if anchors is None:
                logger.warning("Model has no anchors attribute; decode_yolo_predictions expects anchors.")

            # decode at low threshold (captures all preds for sweep)
            dets_batch = decode_yolo_predictions(preds, anchors=anchors, img_size=args.img_size,
                                                 conf_thresh=min_thr_for_decode, iou_thresh=args.nms_iou,
                                                 max_dets=args.max_dets, num_classes=model.num_classes)

            B = images.shape[0]
            for b in range(B):
                img_id = batch["slice_ids"][b]
                n_images += 1

                # Ground-truth
                gt_b = batch["bboxes"][b].cpu().numpy()  # (M,4) yolo-normalized
                gt_l = batch["labels"][b].cpu().numpy()  # (M,) label or -1 padding
                valid = gt_l >= 0
                gt_b = gt_b[valid]; gt_l = gt_l[valid]
                if gt_b.size > 0:
                    n_pos_images += 1
                # Convert to xyxy for IoU
                gt_xyxy = [cxcywh_to_xyxy_norm(bb.tolist()) for bb in gt_b]
                for j in range(len(gt_xyxy)):
                    all_gts[img_id].append((int(gt_l[j]), gt_xyxy[j]))

                # Detections (list of dicts)
                dets = dets_batch[b] if isinstance(dets_batch, list) else dets_batch
                # convert to xyxy + aggregate
                for d in dets:
                    cx, cy, w, h = d["bbox"]
                    xyxy = cxcywh_to_xyxy_norm([cx,cy,w,h])
                    all_preds_raw[img_id].append((int(d["class"]), float(d["confidence"]), xyxy))

                # Optional visualization at display threshold (use first threshold)
                if args.save_vis and (n_images % max(args.vis_every,1) == 0):
                    # Filter for the first threshold in list
                    t_disp = thr_list[0]
                    dets_disp = [d for d in dets if d.get("confidence",0.0) >= t_disp]
                    save_dir_vis = save_dir/"vis"; save_dir_vis.mkdir(parents=True, exist_ok=True)
                    save_visualization(save_dir_vis/f"{img_id}.png", images[b].detach().float().cpu(),
                                       batch["bboxes"][b].detach().cpu(), batch["labels"][b].detach().cpu(),
                                       dets_disp, title=f"{img_id} @ thr={t_disp:.2f}")

            if (bi + 1) % 10 == 0:
                logger.info(f"[{bi+1}/{len(dl)}] processed so far: {n_images} images")

    # Compute global (threshold-independent) mAP
    map_global = compute_map(all_preds_raw, all_gts, num_classes=model.num_classes)
    logging.info(f"GLOBAL mAP@0.50={map_global['mAP50']:.3f}  mAP@[0.50:0.95]={map_global['mAP50_95']:.3f}")

    # Pass 2: for each threshold, compute PR/F1 + thresholded AP
    rows = []
    best_thr = None
    best_metric_val = -1.0

    for thr in thr_list:
        tp_sum = 0; fp_sum = 0; fn_sum = 0
        iou_tps = []

        n_pred_total = 0
        for img_id in all_gts.keys() | all_preds_raw.keys():
            gts = all_gts.get(img_id, [])
            preds = all_preds_raw.get(img_id, [])

            # filter predictions by thr
            preds_f = [(c,s,b) for (c,s,b) in preds if s >= thr]
            # sort descending by score for matching (greedy)
            preds_f.sort(key=lambda x: x[1], reverse=True)

            pred_xyxy = [b for (_,_,b) in preds_f]
            pred_cls   = [c for (c,_,_) in preds_f]
            pred_scores= [s for (_,s,_) in preds_f]
            gt_xyxy = [b for (c,b) in gts]
            gt_cls  = [c for (c,b) in gts]

            n_pred_total += len(pred_xyxy)

            if len(pred_xyxy) == 0 and len(gt_xyxy) == 0:
                continue
            tp, fp, fn, ious = greedy_match_dets_to_gt(pred_xyxy, pred_cls, pred_scores, gt_xyxy, gt_cls, iou_thr=args.match_iou)
            tp_sum += sum(tp); fp_sum += sum(fp); fn_sum += fn
            for i,t in enumerate(tp):
                if t == 1:
                    iou_tps.append(ious[i])

        prec = tp_sum / max(tp_sum + fp_sum, 1e-12)
        rec  = tp_sum / max(tp_sum + fn_sum, 1e-12)
        f1   = 2 * prec * rec / max(prec + rec, 1e-12)
        mean_iou_tp = float(np.mean(iou_tps)) if len(iou_tps) else 0.0

        # Thresholded AP (non-standard)
        map_thr = compute_map_with_threshold(all_preds_raw, all_gts, num_classes=model.num_classes, thr=thr)
        ap50_thr = map_thr["mAP50"]; ap95_thr = map_thr["mAP50_95"]

        row = {
            "threshold": thr,
            "n_images": n_images,
            "n_pos_images": n_pos_images,
            "tp": int(tp_sum), "fp": int(fp_sum), "fn": int(fn_sum),
            "precision": prec, "recall": rec, "f1": f1,
            "mean_iou_tp": mean_iou_tp,
            "ap50_thr": ap50_thr, "ap50_95_thr": ap95_thr,
            "map_global_50": map_global["mAP50"], "map_global_50_95": map_global["mAP50_95"],
            "n_pred": int(n_pred_total)
        }
        rows.append(row)

        key_metric = {"f1": f1, "ap50": ap50_thr, "ap5095": ap95_thr}[args.best_by]
        if key_metric > best_metric_val:
            best_metric_val = key_metric
            best_thr = thr

    # Write metrics.csv
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "threshold","n_images","n_pos_images","n_pred","tp","fp","fn",
            "precision","recall","f1","mean_iou_tp",
            "ap50_thr","ap50_95_thr","map_global_50","map_global_50_95"
        ])
        for r in rows:
            writer.writerow([
                f"{r['threshold']:.5f}", r["n_images"], r["n_pos_images"], r["n_pred"],
                r["tp"], r["fp"], r["fn"],
                f"{r['precision']:.6f}", f"{r['recall']:.6f}", f"{r['f1']:.6f}", f"{r['mean_iou_tp']:.6f}",
                f"{r['ap50_thr']:.6f}", f"{r['ap50_95_thr']:.6f}",
                f"{r['map_global_50']:.6f}", f"{r['map_global_50_95']:.6f}"
            ])

    # Also write PR tuning CSV
    with open(pr_tuning_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # At best threshold, write predictions.jsonl
    with open(pred_path, "w") as out:
        for img_id, preds in all_preds_raw.items():
            # keep only detections >= best_thr
            dets = []
            for (cls_id, score, xyxy) in preds:
                if score >= best_thr:
                    x1,y1,x2,y2 = xyxy
                    cx = (x1+x2)/2.0; cy=(y1+y2)/2.0; w=(x2-x1); h=(y2-y1)
                    dets.append({"bbox":[cx,cy,w,h], "confidence":float(score), "class":int(cls_id)})
            rec = {"image_id": img_id, "detections": dets}
            out.write(json.dumps(rec)+"\n")

    # Save best threshold info
    with open(best_thr_path, "w") as f:
        json.dump({
            "best_by": args.best_by,
            "best_threshold": best_thr,
            "best_metric_value": best_metric_val,
            "map_global_50": map_global["mAP50"],
            "map_global_50_95": map_global["mAP50_95"],
            "model_info": {
                "use_spark_pretrained": getattr(model, 'use_spark_pretrained', False),
                "weights_path": args.weights
            }
        }, f, indent=2)

    dt = time_synchronized() - t0
    logging.info(f"Done. {n_images} images in {dt:.2f}s ({n_images/max(dt,1e-6):.1f} img/s).")
    logging.info(f"GLOBAL: mAP50={map_global['mAP50']:.3f}  mAP50:95={map_global['mAP50_95']:.3f}")
    logging.info(f"Sweep thresholds: {', '.join(f'{t:.2f}' for t in thr_list)}")
    logging.info(f"Best threshold by {args.best_by}: {best_thr:.3f}  (metric={best_metric_val:.4f})")
    logging.info(f"Model type: {'SparK pretrained' if getattr(model, 'use_spark_pretrained', False) else 'Standard'}")
    logging.info(f"metrics.csv written to: {metrics_path}")
    logging.info(f"pr_tuning.csv written to: {pr_tuning_path}")
    logging.info(f"predictions.jsonl (filtered at best threshold) written to: {pred_path}")
    if args.save_vis:
        logging.info(f"Visualizations saved (sampled) under: {save_dir/'vis'}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    inference()