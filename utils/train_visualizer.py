import cv2             
import torch          
import logging
import warnings
import matplotlib 
import numpy as np     
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
matplotlib.use('Agg')  # no display

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, output_dir: str, save_interval: int = 100,
                 conf_thresh: float = 0.5, anchors=None):
        self.vis_dir = Path(output_dir) / 'training_visualizations'
        print(f'Saving visualizations to {self.vis_dir}')
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.save_interval = save_interval
        self.conf_thresh = conf_thresh
        self.batch_count = 0

        if anchors is None:
            anchors = [
                [[16.2, 14.4], [41.1, 33.3], [74.1, 57.6]],      # P3
                [[110.4, 84.0], [146.4, 107.1], [180.3, 132.6]], # P4
                [[226.0, 129.3], [214.8, 188.8], [278.2, 173.3]] # P5
            ]
        self.anchors = anchors
    
    def should_save(self, batch_idx: int) -> bool:
        return batch_idx % self.save_interval == 0
    
    def decode_predictions(
        self,
        predictions,
        img_size=640,
        conf_thresh=0.25,
        iou_thresh=0.45,
        max_dets=50
    ):
        """
        Decode head outputs into YOLO-style detections for the FIRST image in the batch.

        Expected per-scale format:
            (bbox_pred, objectness)
            - bbox_pred:  [B, na*4, H, W]  (channel-first)
            - objectness: [B, na,   H, W]  or [B, na*1, H, W]

        Returns:
            A list of dicts sorted by confidence desc:
                [{'bbox': [cx, cy, w, h], 'confidence': float}, ...]
            where coordinates are normalized to [0,1].
        """
        # Collect boxes and scores across all scales
        all_xyxy = []
        all_scores = []

        # Quick sanity checks
        if not predictions or any(p is None for p in predictions):
            return []

        for scale_idx, scale_pred in enumerate(predictions):
            # Unpack the two-tensor tuple
            if not isinstance(scale_pred, (list, tuple)) or len(scale_pred) != 2:
                logger.warning(f"Scale {scale_idx}: expected a 2-tuple (bbox_pred, objectness); got {type(scale_pred)} with len={getattr(scale_pred, '__len__', 'n/a')}")
                continue

            bbox_pred, objectness = scale_pred

            # Anchor sanity
            if scale_idx >= len(self.anchors):
                logger.warning(f"Scale {scale_idx}: no anchors defined, skipping.")
                continue
            na = len(self.anchors[scale_idx])

            # Shape checks
            if bbox_pred.dim() != 4:
                logger.warning(f"Scale {scale_idx}: bbox_pred should be 4D [B, na*4, H, W], got shape {tuple(bbox_pred.shape)}")
                continue
            if objectness.dim() != 4:
                logger.warning(f"Scale {scale_idx}: objectness should be 4D [B, na, H, W] or [B, na*1, H, W], got {tuple(objectness.shape)}")
                continue

            B, ch_box, H, W = bbox_pred.shape
            if B < 1:
                continue  # nothing to visualize

            # Expect channels = na*4 for bbox
            if ch_box % (4 * na) != 0:
                logger.warning(f"Scale {scale_idx}: bbox channels {ch_box} not divisible by 4*na={4*na}, skipping.")
                continue

            # Objectness channels can be na or na*1; unify to [B, na, H, W]
            B_obj, ch_obj, H_obj, W_obj = objectness.shape
            if (B_obj != B) or (H_obj != H) or (W_obj != W):
                logger.warning(f"Scale {scale_idx}: objectness shape {tuple(objectness.shape)} incompatible with bbox {tuple(bbox_pred.shape)}, skipping.")
                continue

            if ch_obj % na != 0:
                logger.warning(f"Scale {scale_idx}: objectness channels {ch_obj} not divisible by na={na}, skipping.")
                continue

            # Compute stride from feature map width (assumes square img / stride derivation)
            stride = float(img_size) / float(W)

            # Reshape:
            #   bbox_pred -> [B, na, 4, H, W] -> [B, na, H, W, 4]
            #   objectness -> [B, na, H, W]
            bbox_pred = bbox_pred.view(B, na, 4, H, W).permute(0, 1, 3, 4, 2).contiguous()
            objectness = objectness.view(B, na, H, W)

            # Use first image in batch
            box = bbox_pred[0]            # (na, H, W, 4)
            obj = objectness[0]           # (na, H, W)

            # Confidence = sigmoid(objectness)
            conf = torch.sigmoid(obj)     # (na, H, W)

            device = bbox_pred.device
            gy, gx = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )  # (H, W)
            grid = torch.stack((gx, gy), dim=-1).float()  # (H, W, 2)

            # Anchors for this scale (pixels), scaled to feature map units
            scale_anchors = torch.tensor(self.anchors[scale_idx], device=device, dtype=torch.float32) / stride  # (na, 2)

            # Decode:
            #   xy:   sigmoid * 2 - 0.5 (YOLOv5-style)
            #   wh:   (sigmoid * 2)**2 * anchor
            xy = torch.sigmoid(box[..., 0:2]) * 2.0 - 0.5         # (na, H, W, 2)
            wh = (torch.sigmoid(box[..., 2:4]) * 2.0) ** 2        # (na, H, W, 2)
            wh = wh * scale_anchors[:, None, None, :]             # broadcast (na, 1, 1, 2)

            # Convert to normalized centers and sizes
            # (grid is in feature coords; add xy then scale by stride, finally normalize by img_size)
            cx = (xy[..., 0] + grid[..., 0]) * stride / img_size  # (na, H, W)
            cy = (xy[..., 1] + grid[..., 1]) * stride / img_size
            ww = (wh[..., 0] * stride) / img_size
            hh = (wh[..., 1] * stride) / img_size

            # Filter by confidence and small sizes
            mask = (conf > conf_thresh) & (ww > 0.01) & (hh > 0.01)  # (na, H, W)
            if mask.any():
                # Gather selected boxes and scores
                cx = cx[mask]; cy = cy[mask]; ww = ww[mask]; hh = hh[mask]
                scores = conf[mask]

                # Convert to xyxy normalized, clamp to [0,1]
                x1 = torch.clamp(cx - ww / 2.0, 0.0, 1.0)
                y1 = torch.clamp(cy - hh / 2.0, 0.0, 1.0)
                x2 = torch.clamp(cx + ww / 2.0, 0.0, 1.0)
                y2 = torch.clamp(cy + hh / 2.0, 0.0, 1.0)

                all_xyxy.append(torch.stack([x1, y1, x2, y2], dim=-1))  # (N, 4)
                all_scores.append(scores)

        # Nothing kept
        if not all_xyxy:
            return []

        # Concatenate over scales
        xyxy = torch.cat(all_xyxy, dim=0)      # (M, 4), normalized
        scores = torch.cat(all_scores, dim=0)  # (M,)

        # Greedy NMS in pixel space
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

        # Build output
        out = []
        for idx in keep:
            x1, y1, x2, y2 = xyxy[idx].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            ww = (x2 - x1)
            hh = (y2 - y1)
            if ww <= 0 or hh <= 0:
                continue
            out.append({'bbox': [cx, cy, ww, hh], 'confidence': float(scores[idx])})

        return sorted(out, key=lambda d: d['confidence'], reverse=True)

    def load_multimodal_image(self, slice_id):
        """Load all 4 modalities"""
        modalities = ['t1', 't1ce', 't2', 'flair']
        images = []
        
        for modality in modalities:
            possible_paths = [
                self.image_dir / f"{slice_id}_{modality}.png",
                self.image_dir / f"{slice_id}_{modality}.PNG",
            ]
            
            img = None
            for img_path in possible_paths:
                if img_path.exists():
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None and img.size > 0:  
                        break
                    else:
                        img = None
            
            if img is None:
                logger.warning(f"Missing or corrupted {modality} for {slice_id}")
                img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                if img.std() < 1.0:  
                    logger.warning(f"Potentially corrupted {modality} for {slice_id} (low std: {img.std()})")
                
                if img.shape[:2] != (self.img_size, self.img_size):
                    img = cv2.resize(img, (self.img_size, self.img_size))
            
            images.append(img)
        
        multimodal_img = np.stack(images, axis=-1)
        return multimodal_img

    def save_visualization(self, batch_idx: int, epoch: int, images: torch.Tensor, targets: dict, predictions, slice_ids: list):
        try:
            plt.ioff()
            
            img = images[0, 1].detach().cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)
            
            gt_boxes = targets['bboxes'][0].detach().cpu().numpy()
            gt_labels = targets['labels'][0].detach().cpu().numpy()
            
            pred_boxes = self.decode_predictions(
                predictions,
                img_size=img.shape[0],
                conf_thresh=self.conf_thresh,
                iou_thresh=0.45,
                max_dets=50
            )

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(img, cmap='gray')
            
            h, w = img.shape
            
            # Draw GT boxes
            gt_count = 0
            for box, label in zip(gt_boxes, gt_labels):
                if label >= 0:
                    x_center, y_center, width, height = box
                    x1 = (x_center - width/2) * w
                    y1 = (y_center - height/2) * h
                    w_box = width * w
                    h_box = height * h
                    
                    rect = plt.Rectangle((x1, y1), w_box, h_box, 
                                       fill=False, color='lime', linewidth=3, alpha=0.8)
                    ax.add_patch(rect)
                    ax.text(x1, y1-5, 'GT', fontsize=10, color='lime', weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lime', alpha=0.7))
                    gt_count += 1
            
            # Draw pred boxes
            pred_count = 0
            for pred in pred_boxes[:10]:
                x_center, y_center, width, height = pred['bbox']
                confidence = pred['confidence']
                
                if confidence >= self.conf_thresh:
                    x1 = (x_center - width/2) * w
                    y1 = (y_center - height/2) * h
                    w_box = width * w
                    h_box = height * h
                    
                    rect = plt.Rectangle((x1, y1), w_box, h_box, 
                                    fill=False, color='red', linewidth=2, alpha=0.8)
                    ax.add_patch(rect)
                    ax.text(x1, y1-25, f'P:{confidence:.2f}', fontsize=9, color='red', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
                    pred_count += 1
            
            legend_elements = [
                Line2D([0], [0], color='lime', lw=3, label=f'Ground Truth ({gt_count})'),
                Line2D([0], [0], color='red', lw=2, label=f'Predictions ({pred_count})')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
            
            ax.set_title(f'Epoch {epoch}, Batch {batch_idx} | GT: {gt_count}, Pred: {pred_count}', 
                        fontsize=14, weight='bold')
            ax.axis('off')
            
            filename = f'epoch_{epoch:03d}_batch_{batch_idx:04d}_gt{gt_count}_pred{pred_count}.png'
            save_path = self.vis_dir / filename
            fig.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close(fig)
            
            self.batch_count += 1
            
        except Exception as e:
            logger.error(f"Visualization error: {e}")
