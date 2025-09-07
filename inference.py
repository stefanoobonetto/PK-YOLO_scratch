#!/usr/bin/env python3
import json
import csv
import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader

from multimodal_pk_yolo import create_model
from brats_dataset import BraTSDataset, collate_fn
try:
    from utils.utils_inference import (
        parse_args,
        decode_yolo_predictions,
        save_visualization,
        time_synchronized
    )
except Exception:
    from utils_inference import (
        parse_args,
        decode_yolo_predictions,
        save_visualization,
        time_synchronized
    )

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# -----------------------------
# Box utilities (normalized cx,cy,w,h in [0,1])
# -----------------------------
def cxcywh_to_xyxy_norm(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx,cy,w,h) normalized to (x1,y1,x2,y2) normalized."""
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou_xyxy_norm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between two sets of boxes in normalized xyxy. Returns (Na, Nb)."""
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0] if b.ndim else 0))
    # Clamp to [0,1] just in case
    a = a.clamp(0, 1)
    b = b.clamp(0, 1)

    # Areas
    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0))
    area_b = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))

    # Intersection
    lt = torch.max(a[:, None, :2], b[None, :, :2])  # (Na, Nb, 2)
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / (union + 1e-6)
    return iou


def match_detections_to_gt(
    pred_boxes_cxcywh: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes_cxcywh: torch.Tensor,
    iou_threshold: float
) -> Tuple[int, int, int, float]:
    """
    Greedy one-to-one matching of predictions to GT by IoU.
    Returns (tp, fp, fn, mean_iou_of_tp).
    """
    if pred_boxes_cxcywh.numel() == 0:
        tp = 0
        fp = 0
        fn = gt_boxes_cxcywh.shape[0]
        return tp, fp, fn, 0.0

    if gt_boxes_cxcywh.numel() == 0:
        tp = 0
        fp = pred_boxes_cxcywh.shape[0]
        fn = 0
        return tp, fp, fn, 0.0

    # sort predictions by confidence desc
    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes_cxcywh[order]
    pred_scores_sorted = pred_scores[order]

    gt_xyxy = cxcywh_to_xyxy_norm(gt_boxes_cxcywh)
    pred_xyxy = cxcywh_to_xyxy_norm(pred_boxes)

    ious = box_iou_xyxy_norm(gt_xyxy, pred_xyxy)  # (Ng, Np)
    gt_matched = torch.zeros(gt_boxes_cxcywh.shape[0], dtype=torch.bool, device=pred_boxes.device)

    tp = 0
    iou_sum = 0.0
    for p in range(pred_boxes.shape[0]):
        # For this prediction, find the best GT not yet matched
        best_gt = -1
        best_iou = 0.0
        for g in range(gt_boxes_cxcywh.shape[0]):
            if gt_matched[g]:
                continue
            iou = float(ious[g, p].item())
            if iou > best_iou:
                best_iou = iou
                best_gt = g
        if best_gt >= 0 and best_iou >= iou_threshold:
            gt_matched[best_gt] = True
            tp += 1
            iou_sum += best_iou

    fp = pred_boxes.shape[0] - tp
    fn = gt_boxes_cxcywh.shape[0] - tp
    mean_iou = (iou_sum / max(tp, 1)) if tp > 0 else 0.0
    return tp, fp, fn, float(mean_iou)


def safe_div(a: int, b: int) -> float:
    return float(a) / float(b) if b > 0 else 0.0


def inference():
    args = parse_args()

    save_dir = Path(args.save_dir)
    (save_dir / "vis").mkdir(parents=True, exist_ok=True)
    pred_path = save_dir / args.json_name
    metrics_path = save_dir / "metrics.csv"

    ds = BraTSDataset(args.data_dir, split=args.split, img_size=args.img_size, augment=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True,
                    collate_fn=collate_fn)

    device = torch.device(args.device)
    model = create_model(num_classes=1, input_channels=4,
                         pretrained_path=args.weights, device=device.type)
    model.eval()

    anchors = model.anchors.clone() if hasattr(model, "anchors") else torch.tensor([
        [[10, 13], [16, 30], [33, 23]],
        [[30, 61], [62, 45], [59, 119]],
        [[116, 90], [156, 198], [373, 326]],
    ], dtype=torch.float32)

    if args.half and device.type == 'cuda':
        model = model.half()

    n_images = 0
    n_saved_vis = 0
    t0 = time_synchronized()

    # Dataset-level accumulators
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou_sum = 0.0
    total_pred = 0
    total_gt = 0
    total_infer_time_ms = 0.0

    # Open writers
    fjson = open(pred_path, 'w', encoding='utf-8')
    fcsv = open(metrics_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(fcsv)
    csv_writer.writerow([
        "split", "slice_id",
        "n_gt", "n_pred", "tp", "fp", "fn",
        "precision", "recall", "f1", "mean_iou_tp",
        "avg_confidence",
        "infer_time_ms_per_image"
    ])

    with torch.no_grad():
        for bi, batch in enumerate(dl):
            imgs = batch['images'].to(device, non_blocking=True)
            if args.half and device.type == 'cuda':
                imgs = imgs.half()
            else:
                imgs = imgs.float()

            # Forward + timing
            t1 = time_synchronized()
            preds = model(imgs)
            t2 = time_synchronized()
            batch_infer_time_ms = (t2 - t1) * 1000.0
            B = imgs.shape[0]
            per_image_ms = batch_infer_time_ms / max(B, 1)
            total_infer_time_ms += batch_infer_time_ms

            # Decode
            dets_batch = decode_yolo_predictions(
                preds, anchors=anchors, img_size=args.img_size,
                conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh,
                max_dets=args.max_dets, num_classes=model.num_classes
            )

            # Iterate images in batch
            for i in range(B):
                sid = batch['slice_ids'][i]
                dets = dets_batch[i]

                # JSONL record
                record = {
                    "split": args.split,
                    "slice_id": sid,
                    "detections": dets
                }
                fjson.write(json.dumps(record) + "\n")

                # Metrics
                # Ground truth (filter out padded -1 labels)
                gt_boxes = batch['bboxes'][i]
                gt_labels = batch['labels'][i]
                valid = gt_labels >= 0
                gt_boxes = gt_boxes[valid]
                n_gt = int(gt_boxes.shape[0])

                # Predictions
                if len(dets) > 0:
                    pred_boxes = torch.tensor([d["bbox"] for d in dets], dtype=torch.float32, device=device)
                    pred_scores = torch.tensor([d["confidence"] for d in dets], dtype=torch.float32, device=device)
                    avg_conf = float(pred_scores.mean().item())
                else:
                    pred_boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
                    pred_scores = torch.zeros((0,), dtype=torch.float32, device=device)
                    avg_conf = 0.0

                n_pred = int(pred_boxes.shape[0])
                tp, fp, fn, mean_iou = match_detections_to_gt(
                    pred_boxes, pred_scores, gt_boxes.to(device), iou_threshold=args.iou_thresh
                )

                prec = safe_div(tp, tp + fp)
                rec = safe_div(tp, tp + fn)
                f1 = (2 * prec * rec / (prec + rec + 1e-12)) if (prec + rec) > 0 else 0.0

                # Write per-image row
                csv_writer.writerow([
                    args.split, sid,
                    n_gt, n_pred, tp, fp, fn,
                    round(prec, 6), round(rec, 6), round(f1, 6), round(mean_iou, 6),
                    round(avg_conf, 6),
                    round(per_image_ms, 3),
                ])

                # Accumulate
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_iou_sum += (mean_iou * (tp if tp > 0 else 0))
                total_pred += n_pred
                total_gt += n_gt

                # Save visualization periodically (uses GT if available)
                if args.save_vis and (bi % args.save_interval == 0):
                    save_path = save_dir / "vis" / f"{sid}_pred.png"
                    title = f"{args.split} | {sid} | conf>{args.conf_thresh} iouNMS={args.iou_thresh}"
                    save_visualization(save_path, imgs[i].float().cpu(), gt_boxes, gt_labels, dets, title)
                    n_saved_vis += 1

            n_images += B

            if (bi + 1) % 10 == 0:
                logger.info(f"[{bi+1}/{len(dl)}] processed {n_images} images")

    # Dataset summary metrics
    overall_prec = safe_div(total_tp, total_tp + total_fp)
    overall_rec = safe_div(total_tp, total_tp + total_fn)
    overall_f1 = (2 * overall_prec * overall_rec / (overall_prec + overall_rec + 1e-12)) if (overall_prec + overall_rec) > 0 else 0.0
    overall_mean_iou_tp = (total_iou_sum / max(total_tp, 1)) if total_tp > 0 else 0.0
    avg_ms_per_image = total_infer_time_ms / max(n_images, 1)

    # Write summary row
    csv_writer.writerow([])
    csv_writer.writerow([
        "SUMMARY", "ALL",
        total_gt, total_pred, total_tp, total_fp, total_fn,
        round(overall_prec, 6), round(overall_rec, 6), round(overall_f1, 6),
        round(overall_mean_iou_tp, 6),
        "",  # avg_confidence not meaningful aggregated
        round(avg_ms_per_image, 3),
    ])

    fjson.close()
    fcsv.close()

    dt = time_synchronized() - t0
    logger.info(f"Done. Processed {n_images} images in {dt:.2f}s "
                f"({n_images / max(dt,1e-6):.1f} img/s).")
    logger.info(f"Predictions written to: {pred_path}")
    logger.info(f"Metrics CSV written to: {metrics_path}")
    if args.save_vis:
        logger.info(f"Visualizations saved in: {save_dir / 'vis'}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    inference()
