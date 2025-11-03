import cv2
import torch
import logging
import warnings
import matplotlib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
matplotlib.use('Agg')  # headless rendering

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Visualizer:
    def __init__(
        self,
        output_dir: str,
        save_interval: int = 100,
        conf_thresh: float = 0.5,
        anchors=None,
        img_size: int = 640,
        image_dir: str | None = None,
    ):
        self.vis_dir = Path(output_dir) / 'training_visualizations'
        print(f'Saving visualizations to {self.vis_dir}')
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.save_interval = int(save_interval)
        self.conf_thresh = float(conf_thresh)
        self.batch_count = 0

        self.img_size = int(img_size)
        self.image_dir = Path(image_dir) if image_dir is not None else (Path(output_dir) / "images")

        # default anchors (pixel units at base image size)
        if anchors is None:
            anchors = [
                [[  4,   4], [  8,   7], [ 13,  11]],  # P2 (8x)
                [[ 19,  17], [ 28,  24], [ 41,  35]],  # P3 (16x)
                [[ 59,  52], [ 87,  74], [122,  98]],  # P4 (32x)
                [[165, 141], [234, 187], [340, 280]],  # P5 (64x)
            ]
        self.anchors = anchors

    def should_save(self, batch_idx: int) -> bool:
        return (batch_idx % self.save_interval) == 0

    @torch.no_grad()
    def decode_predictions(
        self,
        predictions,
        img_size: int | None = None,
        conf_thresh: float | None = None,
        iou_thresh: float = 0.45,
        max_dets: int = 50,
    ):
        """
        Decode head outputs into YOLO-style detections for the FIRST image in the batch.

        Expected per-scale format: (bbox_pred, objectness)
          - bbox_pred:  [B, na*4, H, W]  (tx, ty, tw, th)
          - objectness: [B, na,   H, W]  or [B, na*1, H, W]

        Returns (sorted by confidence desc):
          [{'bbox': [cx, cy, w, h], 'confidence': float}, ...]  # normalized [0,1]
        """
        img_size = int(img_size or self.img_size)
        conf_thresh = float(conf_thresh if conf_thresh is not None else self.conf_thresh)

        all_xyxy, all_scores = [], []

        if not predictions or any(p is None for p in predictions):
            return []

        for scale_idx, scale_pred in enumerate(predictions):
            if not isinstance(scale_pred, (list, tuple)) or len(scale_pred) != 2:
                logger.warning(f"Scale {scale_idx}: expected a 2-tuple (bbox_pred, objectness); got {type(scale_pred)}")
                continue

            bbox_pred, objectness = scale_pred

            if scale_idx >= len(self.anchors):
                logger.warning(f"Scale {scale_idx}: no anchors defined, skipping.")
                continue
            na = len(self.anchors[scale_idx])

            if bbox_pred.dim() != 4:
                logger.warning(f"Scale {scale_idx}: bbox_pred should be [B, na*4, H, W], got {tuple(bbox_pred.shape)}")
                continue
            if objectness.dim() != 4:
                logger.warning(f"Scale {scale_idx}: objectness should be [B, na, H, W] or [B, na*1, H, W], got {tuple(objectness.shape)}")
                continue

            B, ch_box, H, W = bbox_pred.shape
            if B < 1:
                continue
            if ch_box % (4 * na) != 0:
                logger.warning(f"Scale {scale_idx}: bbox channels {ch_box} not divisible by 4*na={4*na}, skipping.")
                continue

            B_obj, ch_obj, H_obj, W_obj = objectness.shape
            if (B_obj != B) or (H_obj != H) or (W_obj != W) or (ch_obj % na != 0):
                logger.warning(f"Scale {scale_idx}: objectness shape {tuple(objectness.shape)} incompatible with bbox {tuple(bbox_pred.shape)}, skipping.")
                continue

            # stride in pixels per grid cell (assumes square)
            stride = float(img_size) / float(W)

            # reshape to [B, na, H, W, *]
            bbox_pred = bbox_pred.view(B, na, 4, H, W).permute(0, 1, 3, 4, 2).contiguous()
            objectness = objectness.view(B, na, H, W)

            # take first image
            box = bbox_pred[0]      # [na, H, W, 4]
            obj = objectness[0]     # [na, H, W]
            conf = torch.sigmoid(obj)

            device = bbox_pred.device
            gy, gx = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            grid = torch.stack((gx, gy), dim=-1).float()  # (H, W, 2)

            # anchors (pixel) -> grid units
            scale_anchors = torch.tensor(self.anchors[scale_idx], device=device, dtype=torch.float32) / stride  # (na, 2)

            # decode
            xy = torch.sigmoid(box[..., 0:2]) * 2.0 - 0.5            # (na, H, W, 2)
            wh = (torch.sigmoid(box[..., 2:4]) * 2.0) ** 2           # (na, H, W, 2)
            wh = wh * scale_anchors[:, None, None, :]                # (na, H, W, 2)

            cx = (xy[..., 0] + grid[..., 0]) * stride / img_size     # normalized
            cy = (xy[..., 1] + grid[..., 1]) * stride / img_size
            ww = (wh[..., 0] * stride) / img_size
            hh = (wh[..., 1] * stride) / img_size

            # pre-filter
            mask = (conf > conf_thresh) & (ww > 0.01) & (hh > 0.01)
            if mask.any():
                cx = cx[mask]; cy = cy[mask]; ww = ww[mask]; hh = hh[mask]
                scores = conf[mask]

                x1 = (cx - ww * 0.5).clamp(0.0, 1.0)
                y1 = (cy - hh * 0.5).clamp(0.0, 1.0)
                x2 = (cx + ww * 0.5).clamp(0.0, 1.0)
                y2 = (cy + hh * 0.5).clamp(0.0, 1.0)

                all_xyxy.append(torch.stack([x1, y1, x2, y2], dim=-1))
                all_scores.append(scores)

        if not all_xyxy:
            return []

        xyxy = torch.cat(all_xyxy, dim=0)
        scores = torch.cat(all_scores, dim=0)

        # greedy NMS
        xyxy_pix = xyxy * float(img_size)
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i.item())
            if order.numel() == 1:
                break
            xx1 = torch.maximum(xyxy_pix[i, 0], xyxy_pix[order[1:], 0])
            yy1 = torch.maximum(xyxy_pix[i, 1], xyxy_pix[order[1:], 1])
            xx2 = torch.minimum(xyxy_pix[i, 2], xyxy_pix[order[1:], 2])
            yy2 = torch.minimum(xyxy_pix[i, 3], xyxy_pix[order[1:], 3])
            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            area_i = (xyxy_pix[i, 2] - xyxy_pix[i, 0]).clamp(min=0) * (xyxy_pix[i, 3] - xyxy_pix[i, 1]).clamp(min=0)
            area_r = (xyxy_pix[order[1:], 2] - xyxy_pix[order[1:], 0]).clamp(min=0) * (xyxy_pix[order[1:], 3] - xyxy_pix[order[1:], 1]).clamp(min=0)
            iou = inter / (area_i + area_r - inter + 1e-6)
            order = order[1:][iou <= iou_thresh]

        keep = keep[:max_dets]

        out = []
        for idx in keep:
            x1, y1, x2, y2 = xyxy[idx].tolist()
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            ww = (x2 - x1)
            hh = (y2 - y1)
            if ww <= 0 or hh <= 0:
                continue
            out.append({'bbox': [cx, cy, ww, hh], 'confidence': float(scores[idx])})

        return sorted(out, key=lambda d: d['confidence'], reverse=True)

    def load_multimodal_image(self, slice_id: str):
        """
        Loads 4 modalities from self.image_dir. Requires self.image_dir & self.img_size.
        Not used by save_visualization() but kept for convenience.
        """
        modalities = ['t1', 't1ce', 't2', 'flair']
        images = []

        for modality in modalities:
            img = None
            for ext in ("png", "PNG"):
                p = self.image_dir / f"{slice_id}_{modality}.{ext}"
                if p.exists():
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    break
            if img is None or img.size == 0:
                logger.warning(f"Missing/corrupted {modality} for {slice_id}, using zeros.")
                img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                if img.std() < 1.0:
                    logger.warning(f"Potentially corrupted {modality} for {slice_id} (low std: {img.std():.3f})")
                if img.shape[:2] != (self.img_size, self.img_size):
                    img = cv2.resize(img, (self.img_size, self.img_size))
            images.append(img)

        return np.stack(images, axis=-1)  # (H,W,4)

    @torch.no_grad()
    def save_visualization(self, batch_idx: int, epoch: int, images: torch.Tensor, targets: dict, predictions, slice_ids: list):
        try:
            plt.ioff()

            # Use T1ce (index 1); switch if you prefer another channel
            img_t = images[0, 1].detach().float().cpu()
            imn = img_t
            denom = (imn.max() - imn.min()).clamp_min(1e-6)
            img = ((imn - imn.min()) / denom * 255.0).byte().numpy()  # (H,W)

            # Ground-truth: filter valid boxes
            gt_boxes = targets['bboxes'][0].detach().cpu().float().numpy()  # [M,4] in [0,1]
            gt_labels = targets['labels'][0].detach().cpu().long().numpy()  # [M]
            H, W = img.shape

            valid_mask = (gt_boxes[:, 2] > 0) & (gt_boxes[:, 3] > 0)  # w,h > 0
            gt_boxes = gt_boxes[valid_mask]
            gt_labels = gt_labels[valid_mask]

            # Predictions for first image (normalized coords)
            preds = self.decode_predictions(
                predictions, img_size=H, conf_thresh=self.conf_thresh, iou_thresh=0.45, max_dets=50
            )

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(img, cmap='gray')

            # draw GT (lime)
            gt_count = 0
            for box, label in zip(gt_boxes, gt_labels):
                cx, cy, ww, hh = box
                if ww <= 0 or hh <= 0:
                    continue
                x1 = (cx - ww * 0.5) * W
                y1 = (cy - hh * 0.5) * H
                rect = plt.Rectangle((x1, y1), ww * W, hh * H,
                                     fill=False, color='lime', linewidth=2.5, alpha=0.9)
                ax.add_patch(rect)
                ax.text(x1, max(0, y1 - 6), 'GT', fontsize=10, color='black',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='lime', alpha=0.8))
                gt_count += 1

            # draw predictions (red)
            pred_count = 0
            for pred in preds[:10]:
                cx, cy, ww, hh = pred['bbox']
                conf = pred['confidence']
                x1 = (cx - ww * 0.5) * W
                y1 = (cy - hh * 0.5) * H
                rect = plt.Rectangle((x1, y1), ww * W, hh * H,
                                     fill=False, color='red', linewidth=2, alpha=0.9)
                ax.add_patch(rect)
                ax.text(x1, max(0, y1 - 16), f'P:{conf:.2f}', fontsize=9, color='white',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.8))
                pred_count += 1

            legend_elements = [
                Line2D([0], [0], color='lime', lw=2.5, label=f'Ground Truth ({gt_count})'),
                Line2D([0], [0], color='red',  lw=2.0, label=f'Predictions ({pred_count})'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

            ax.set_title(f'Epoch {epoch}, Batch {batch_idx} | GT: {gt_count}, Pred: {pred_count}',
                         fontsize=14, weight='bold')
            ax.axis('off')

            filename = f'epoch_{epoch:03d}_batch_{batch_idx:04d}_gt{gt_count}_pred{pred_count}.png'
            save_path = self.vis_dir / filename
            fig.savefig(save_path, bbox_inches='tight', dpi=110)
            plt.close(fig)

            self.batch_count += 1

        except Exception as e:
            logger.error(f"Visualization error: {e}")
