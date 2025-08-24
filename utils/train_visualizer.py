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
    def __init__(self, output_dir: str, save_interval: int = 100, conf_thresh: float = 0.05):
        self.vis_dir = Path(output_dir) / 'training_visualizations'
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_interval = save_interval
        self.conf_thresh = conf_thresh
        self.batch_count = 0
    
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
        # Decode model outputs to normalized YOLO-style [cx, cy, w, h] + confidence.
        # Returns a list of dicts: {'bbox':[cx,cy,w,h], 'confidence':float}
        
        anchors = [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]],
        ]

        all_xyxy = []
        all_scores = []

        for scale_idx, (cls_score, bbox_pred, objectness) in enumerate(predictions):
            _, _, h, w = cls_score.shape
            stride = float(img_size) / float(w)

            if scale_idx >= len(anchors):
                continue

            B = cls_score.shape[0]
            na = 3
            cls_score = cls_score.view(B, na, 1, h, w).permute(0, 1, 3, 4, 2)  # (B,3,H,W,1)
            bbox_pred = bbox_pred.view(B, na, 4, h, w).permute(0, 1, 3, 4, 2)  # (B,3,H,W,4)
            objectness = objectness.view(B, na, h, w)                            # (B,3,H,W)

            # prob
            cls_prob = torch.sigmoid(cls_score)[0, ..., 0]    # (3,H,W)
            obj_prob = torch.sigmoid(objectness)[0]           # (3,H,W)
            conf = (cls_prob * obj_prob)                      # (3,H,W)

            device = cls_score.device
            gy, gx = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing='ij'
            )  # (H,W)
            
            grid = torch.stack((gx, gy), dim=-1).float()      # (H,W,2)

            scale_anchors = torch.tensor(anchors[scale_idx], device=device, dtype=torch.float32) / stride  # (3,2)

            box = bbox_pred[0]                                # (3,H,W,4)
            xy = torch.sigmoid(box[..., 0:2]) * 2.0 - 0.5     # (3,H,W,2)
            wh = (torch.sigmoid(box[..., 2:4]) * 2.0) ** 2    # (3,H,W,2)
            wh = wh * scale_anchors[:, None, None, :]         

            # centers in pixels / normalized
            cx = (xy[..., 0] + grid[..., 0]) * stride / img_size  # (3,H,W)
            cy = (xy[..., 1] + grid[..., 1]) * stride / img_size  # (3,H,W)
            ww = (wh[..., 0] * stride) / img_size
            hh = (wh[..., 1] * stride) / img_size

            mask = (conf > conf_thresh) & (ww > 0.01) & (hh > 0.01)
            if mask.any():
                cx = cx[mask]; cy = cy[mask]; ww = ww[mask]; hh = hh[mask]
                scores = conf[mask]

                # convert to xyxy (normalized), clamp to [0,1]
                x1 = torch.clamp(cx - ww / 2.0, 0.0, 1.0)
                y1 = torch.clamp(cy - hh / 2.0, 0.0, 1.0)
                x2 = torch.clamp(cx + ww / 2.0, 0.0, 1.0)
                y2 = torch.clamp(cy + hh / 2.0, 0.0, 1.0)

                all_xyxy.append(torch.stack([x1, y1, x2, y2], dim=-1))
                all_scores.append(scores)

        if not all_xyxy:
            return []

        xyxy = torch.cat(all_xyxy, dim=0)
        scores = torch.cat(all_scores, dim=0)

        xyxy_pix = xyxy * float(img_size)
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i.item())
            if order.numel() == 1:
                break
            # IoU vs remaining boxes
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
