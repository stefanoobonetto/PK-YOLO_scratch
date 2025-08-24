import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import logging
from pathlib import Path
from typing import List, Dict, Any

import torch
torch.set_num_threads(1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("inference")

def build_grid(h: int, w: int, device: torch.device):
    gy, gx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij'
    )  # (H,W)
    grid = torch.stack((gx, gy), dim=-1).float()  # (H,W,2) with (x,y)
    return grid

def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return x.sigmoid()

def nms_xyxy(xyxy: torch.Tensor, scores: torch.Tensor, iou_thresh: float, max_dets: int):

    if xyxy.numel() == 0:
        return []

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if len(keep) >= max_dets or order.numel() == 1:
            break
        xx1 = torch.maximum(xyxy[i, 0], xyxy[order[1:], 0])
        yy1 = torch.maximum(xyxy[i, 1], xyxy[order[1:], 1])
        xx2 = torch.minimum(xyxy[i, 2], xyxy[order[1:], 2])
        yy2 = torch.minimum(xyxy[i, 3], xyxy[order[1:], 3])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        area_i = (xyxy[i, 2] - xyxy[i, 0]).clamp(min=0) * (xyxy[i, 3] - xyxy[i, 1]).clamp(min=0)
        area_r = (xyxy[order[1:], 2] - xyxy[order[1:], 0]).clamp(min=0) * (xyxy[order[1:], 3] - xyxy[order[1:], 1]).clamp(min=0)
        iou = inter / (area_i + area_r - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep

def decode_yolo_predictions(
    predictions,
    anchors: torch.Tensor,
    img_size: int,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_dets: int = 50,
    num_classes: int = 1
) -> List[List[Dict[str, Any]]]:
    """
    Decode raw head outputs into per-image detections.
    Args:
        predictions: list of tuples [(cls_score, bbox_pred, objectness), ...] for 3 scales
            - cls_score shape: (B, A*C, H, W)
            - bbox_pred shape: (B, A*4, H, W)
            - objectness shape: (B, A, H, W)
        anchors: (3,3,2) tensor in pixels for P3,P4,P5 (matching predictions order)
        img_size: int (assumes square after preprocessing)
    Returns:
        detections: list of length B; each is a list of dicts:
            {'bbox':[cx,cy,w,h] (normalized), 'confidence':float, 'class':int}
    """
    device = predictions[0][0].device
    B = predictions[0][0].shape[0]
    all_img_dets = [[] for _ in range(B)]

    for scale_idx, (cls_score, bbox_pred, objectness) in enumerate(predictions):
        B, ac, h, w = cls_score.shape
        A = objectness.shape[1]
        C = ac // A
        stride = float(img_size) / float(w)

        # reshape
        cls_score = cls_score.view(B, A, C, h, w)       # (B,A,C,H,W)
        bbox_pred = bbox_pred.view(B, A, 4, h, w)       # (B,A,4,H,W)
        objectness = objectness                          # (B,A,H,W)

        # probs
        cls_prob = sigmoid(cls_score)                    # (B,A,C,H,W)
        obj_prob = sigmoid(objectness).unsqueeze(2)      # (B,A,1,H,W)

        # If multi-class, take best class; for single-class this is trivial
        conf, cls_idx = (cls_prob * obj_prob).max(dim=2)  # (B,A,H,W), (B,A,H,W)

        # grid and anchors
        grid = build_grid(h, w, device)                 # (H,W,2)
        # scale anchors for this level (normalize to stride)
        scale_anchors = (anchors[scale_idx].to(device).float() / stride)  # (A,2)

        # decode xywh
        xy = sigmoid(bbox_pred[:, :, 0:2, ...]).permute(0,1,3,4,2) * 2.0 - 0.5  # (B,A,H,W,2)
        wh = sigmoid(bbox_pred[:, :, 2:4, ...]).permute(0,1,3,4,2)
        wh = (wh * 2.0)**2 * scale_anchors[None, :, None, None, :]             # (B,A,H,W,2)

        # centers in pixels (normalize later)
        cx = (xy[..., 0] + grid[..., 0]) * stride        # (B,A,H,W)
        cy = (xy[..., 1] + grid[..., 1]) * stride        # (B,A,H,W)
        ww = wh[..., 0] * stride
        hh = wh[..., 1] * stride

        # filter
        mask = (conf > conf_thresh) & (ww > 1.0) & (hh > 1.0)  # pixel size threshold 1
        if not mask.any():
            continue

        # gather masked boxes per image
        for b in range(B):
            mb = mask[b]  # (A,H,W)
            if not mb.any():
                continue
            cx_b = cx[b][mb]; cy_b = cy[b][mb]; ww_b = ww[b][mb]; hh_b = hh[b][mb]
            scores_b = conf[b][mb]
            cls_b = cls_idx[b][mb]

            x1 = (cx_b - ww_b/2.0)
            y1 = (cy_b - hh_b/2.0)
            x2 = (cx_b + ww_b/2.0)
            y2 = (cy_b + hh_b/2.0)

            xyxy_pix = torch.stack([x1, y1, x2, y2], dim=-1)
            # clamp to image boundaries
            xyxy_pix[..., 0::2] = xyxy_pix[..., 0::2].clamp(0, img_size)
            xyxy_pix[..., 1::2] = xyxy_pix[..., 1::2].clamp(0, img_size)

            # NMS per image across all scales collected later
            # For simplicity accumulate now; we'll NMS across scales after loop.
            all_img_dets[b].append((xyxy_pix, scores_b, cls_b))

    # Concatenate per image across scales and NMS
    final = []
    for b in range(B):
        if len(all_img_dets[b]) == 0:
            final.append([])
            continue
        xyxy = torch.cat([t[0] for t in all_img_dets[b]], dim=0)
        scores = torch.cat([t[1] for t in all_img_dets[b]], dim=0)
        cls_ids = torch.cat([t[2] for t in all_img_dets[b]], dim=0)

        keep = nms_xyxy(xyxy, scores, iou_thresh=iou_thresh, max_dets=max_dets)
        dets = []
        for idx in keep:
            x1, y1, x2, y2 = xyxy[idx].tolist()
            cx = (x1 + x2) / 2.0 / img_size
            cy = (y1 + y2) / 2.0 / img_size
            ww = (x2 - x1) / img_size
            hh = (y2 - y1) / img_size
            dets.append({
                "bbox": [cx, cy, ww, hh],   # normalized
                "confidence": float(scores[idx].item()),
                "class": int(cls_ids[idx].item())
            })
        final.append(dets)
    return final

def save_visualization(
    save_path: Path,
    image_tensor: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    detections: List[Dict[str, Any]],
    title: str = ""
):
    # Save image with GT and predicted boxes 

    img = image_tensor.detach().cpu()
    if img.dim() == 3:
        # Choose channel 1 as in training visualizer for MRI (T1ce) or mean across channels
        img = img[1] if img.shape[0] > 1 else img[0]
    img = img.numpy()
    # Normalize for display
    vmin, vmax = float(img.min()), float(img.max())
    img_vis = (img - vmin) / (vmax - vmin + 1e-6)

    h, w = img_vis.shape[-2], img_vis.shape[-1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_vis, cmap='gray')

    # GT
    gt_count = 0
    if gt_boxes is not None and gt_labels is not None:
        gt_boxes = gt_boxes.detach().cpu()
        gt_labels = gt_labels.detach().cpu()
        for box, label in zip(gt_boxes, gt_labels):
            if int(label.item()) < 0:
                continue
            x, y, bw, bh = box.tolist()
            x1 = (x - bw/2) * w
            y1 = (y - bh/2) * h
            rect = plt.Rectangle((x1, y1), bw*w, bh*h, fill=False, linewidth=2, color='lime', alpha=0.9)
            ax.add_patch(rect)
            ax.text(x1, max(0, y1-4), "GT", fontsize=9, color='lime',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.4))
            gt_count += 1

    # Predictions
    pred_count = 0
    for det in detections[:50]:
        cx, cy, bw, bh = det["bbox"]
        conf = det["confidence"]
        x1 = (cx - bw/2) * w
        y1 = (cy - bh/2) * h
        rect = plt.Rectangle((x1, y1), bw*w, bh*h, fill=False, linewidth=2, color='red', alpha=0.9)
        ax.add_patch(rect)
        ax.text(x1, max(0, y1-12), f"P:{conf:.2f}", fontsize=8, color='red',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.4))
        pred_count += 1

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lime', lw=2, label=f'GT ({gt_count})'),
        Line2D([0], [0], color='red', lw=2, label=f'Pred ({pred_count})')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax.set_title(title or f"GT {gt_count} / Pred {pred_count}")
    ax.axis('off')

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), bbox_inches='tight', dpi=120)
    plt.close(fig)

def parse_args():
    import argparse
    p = argparse.ArgumentParser("MultimodalPKYOLO Inference")
    p.add_argument('--data_dir', type=str, required=True, help='Root of dataset with split subfolders')
    p.add_argument('--split', type=str, default='val', choices=['val', 'test', 'train'], help='Dataset split')
    p.add_argument('--weights', type=str, required=True, help='Path to checkpoint .pt/.pth')
    p.add_argument('--img_size', type=int, default=640, help='Input resolution (assumed square)')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--conf_thresh', type=float, default=0.25)
    p.add_argument('--iou_thresh', type=float, default=0.45)
    p.add_argument('--max_dets', type=int, default=50)
    p.add_argument('--save_dir', type=str, default='./runs/inference/exp')
    p.add_argument('--save_vis', action='store_true', help='Save overlay images')
    p.add_argument('--save_interval', type=int, default=100, help='Save every N batches for overlays')
    p.add_argument('--json_name', type=str, default='predictions.jsonl')
    p.add_argument('--half', action='store_true', help='Use fp16 (CUDA only)')
    return p.parse_args()