import cv2
import torch
import logging
import warnings
import matplotlib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.use("Agg")  # headless
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Visualizer:
    def __init__(
        self,
        output_dir: str,
        img_size: int = 640,
        image_dir: str | None = None,
        save_interval: int = 100,
        conf_thresh: float = 0.5,
        anchors=None,
    ):
        """Simple training visualizer for 4-channel (t1, t1ce, t2, flair) inputs."""
        self.vis_dir = Path(output_dir) / "training_visualizations"
        print(f"Saving visualizations to {self.vis_dir}")
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.img_size = int(img_size)
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.save_interval = int(save_interval)
        self.conf_thresh = float(conf_thresh)
        self.batch_count = 0

        if anchors is None:
            anchors = [
                [[16.2, 14.4], [41.1, 33.3], [74.1, 57.6]],       # P3
                [[110.4, 84.0], [146.4, 107.1], [180.3, 132.6]],  # P4
                [[226.0, 129.3], [214.8, 188.8], [278.2, 173.3]], # P5
            ]
        self.anchors = anchors

    def should_save(self, batch_idx: int) -> bool:
        return (batch_idx % self.save_interval) == 0

    @torch.no_grad()
    def decode_predictions(
        self,
        predictions,
        img_size=640,
        conf_thresh=0.25,
        iou_thresh=0.45,
        max_dets=50,
    ):
        """
        Decode head outputs into YOLO-style detections for the FIRST image in the batch.

        Expected per-scale format:
          (bbox_pred, objectness)
            - bbox_pred:  [B, na*4, H, W]
            - objectness: [B, na,   H, W] or [B, na*1, H, W]

        Returns: list[{'bbox': [cx, cy, w, h], 'confidence': float}] (normalized to [0,1])
        """
        all_xyxy, all_scores = [], []

        if not predictions or any(p is None for p in predictions):
            return []

        for scale_idx, scale_pred in enumerate(predictions):
            if not isinstance(scale_pred, (list, tuple)) or len(scale_pred) != 2:
                logger.warning(f"Scale {scale_idx}: expected (bbox_pred, objectness), got {type(scale_pred)}")
                continue

            bbox_pred, objectness = scale_pred

            if scale_idx >= len(self.anchors):
                logger.warning(f"Scale {scale_idx}: no anchors defined, skipping.")
                continue
            na = len(self.anchors[scale_idx])

            if bbox_pred.dim() != 4 or objectness.dim() != 4:
                logger.warning(f"Scale {scale_idx}: bad dims, bbox {tuple(bbox_pred.shape)}, obj {tuple(objectness.shape)}")
                continue

            B, ch_box, H, W = bbox_pred.shape
            if B < 1:
                continue
            if ch_box % (4 * na) != 0:
                logger.warning(f"Scale {scale_idx}: bbox channels {ch_box} not divisible by 4*na.")
                continue

            B_obj, ch_obj, H_obj, W_obj = objectness.shape
            if (B_obj != B) or (H_obj != H) or (W_obj != W) or (ch_obj % na != 0):
                logger.warning(f"Scale {scale_idx}: obj shape {tuple(objectness.shape)} mismatched with bbox {tuple(bbox_pred.shape)}")
                continue

            stride = float(img_size) / float(W)

            bbox_pred = bbox_pred.view(B, na, 4, H, W).permute(0, 1, 3, 4, 2).contiguous()
            objectness = objectness.view(B, na, H, W)

            box = bbox_pred[0]                      # (na, H, W, 4)
            obj = objectness[0]                     # (na, H, W)
            conf = torch.sigmoid(obj)               # (na, H, W)

            device = bbox_pred.device
            gy, gx = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij",
            )
            grid = torch.stack((gx, gy), dim=-1).float()  # (H, W, 2)

            scale_anchors = torch.tensor(self.anchors[scale_idx],
                                         device=device, dtype=torch.float32) / stride  # (na,2)

            xy = torch.sigmoid(box[..., 0:2]) * 2.0 - 0.5
            wh = (torch.sigmoid(box[..., 2:4]) * 2.0) ** 2
            wh = wh * scale_anchors[:, None, None, :]

            cx = (xy[..., 0] + grid[..., 0]) * stride / img_size
            cy = (xy[..., 1] + grid[..., 1]) * stride / img_size
            ww = (wh[..., 0] * stride) / img_size
            hh = (wh[..., 1] * stride) / img_size

            mask = (conf > conf_thresh) & (ww > 0.01) & (hh > 0.01)
            if mask.any():
                cx, cy, ww, hh = cx[mask], cy[mask], ww[mask], hh[mask]
                scores = conf[mask]

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
            out.append({"bbox": [cx, cy, ww, hh], "confidence": float(scores[idx])})

        return sorted(out, key=lambda d: d["confidence"], reverse=True)

    def load_multimodal_image(self, slice_id: str) -> np.ndarray:
        """Load all 4 modalities from disk (requires image_dir)."""
        if self.image_dir is None:
            raise ValueError("image_dir was not provided to Visualizer.")
        modalities = ["t1", "t1ce", "t2", "flair"]
        imgs = []

        for modality in modalities:
            candidates = [
                self.image_dir / f"{slice_id}_{modality}.png",
                self.image_dir / f"{slice_id}_{modality}.PNG",
            ]
            img = None
            for p in candidates:
                if p.exists():
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is not None and img.size > 0:
                        break
                    img = None

            if img is None:
                logger.warning(f"Missing or corrupted {modality} for {slice_id}")
                img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                if img.shape[:2] != (self.img_size, self.img_size):
                    img = cv2.resize(img, (self.img_size, self.img_size))
                if img.std() < 1.0:
                    logger.warning(f"Potentially corrupted {modality} for {slice_id} (low std: {img.std():.4f})")

            imgs.append(img)

        return np.stack(imgs, axis=-1)  # [H,W,4]

    @torch.no_grad()
    def save_visualization(
        self,
        batch_idx: int,
        epoch: int,
        images: torch.Tensor,      # [B,C,H,W] with C=4
        targets: dict,
        predictions,
        slice_ids: list | None = None,
    ):
        """
        Render composite image + GT + predictions for the first item in the batch.
        Uses robust per-channel min-max normalization and max-projection to avoid
        black frames when a single modality is empty.
        """
        try:
            plt.ioff()

            # ----- robust composite across channels -----
            # images: [B,C,H,W]; take first sample
            img_t = images[0].detach().float().cpu()             # [C,H,W]
            if img_t.ndim != 3:
                raise ValueError(f"Expected images[0] with shape [C,H,W], got {tuple(img_t.shape)}")

            C, H, W = img_t.shape
            mins = img_t.view(C, -1).min(dim=1).values.view(C, 1, 1)
            maxs = img_t.view(C, -1).max(dim=1).values.view(C, 1, 1)
            norm = (img_t - mins) / (maxs - mins + 1e-6)         # [C,H,W] in [0,1]
            comp = norm.max(dim=0).values                        # [H,W]
            img = (comp.clamp(0, 1) * 255.0).byte().numpy()      # uint8 for imshow
            empty_slice = (comp.max().item() < 1e-6)

            # ----- ground truth -----
            gt_boxes = targets.get("bboxes", None)
            gt_labels = targets.get("labels", None)
            if gt_boxes is None or gt_labels is None:
                gt_boxes_np = np.zeros((0, 4), dtype=np.float32)
                gt_labels_np = np.zeros((0,), dtype=np.int64)
            else:
                gt_boxes_np = gt_boxes[0].detach().cpu().numpy()
                gt_labels_np = gt_labels[0].detach().cpu().numpy()

            # ----- predictions -> normalized bboxes -----
            pred_boxes = self.decode_predictions(
                predictions,
                img_size=img.shape[0],
                conf_thresh=self.conf_thresh,
                iou_thresh=0.45,
                max_dets=50,
            )

            # ----- draw -----
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(img, cmap="gray")
            if empty_slice:
                ax.text(10, 20, "EMPTY/FAILED IMAGE", color="red", fontsize=12)

            h, w = img.shape
            gt_count = 0
            for box, label in zip(gt_boxes_np, gt_labels_np):
                if int(label) < 0:
                    continue
                x_center, y_center, width, height = box
                x1 = (x_center - width / 2.0) * w
                y1 = (y_center - height / 2.0) * h
                w_box = width * w
                h_box = height * h

                rect = plt.Rectangle((x1, y1), w_box, h_box,
                                     fill=False, color="lime", linewidth=3, alpha=0.8)
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, "GT", fontsize=10, color="lime", weight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lime", alpha=0.7))
                gt_count += 1

            pred_count = 0
            for pred in pred_boxes[:10]:
                x_center, y_center, width, height = pred["bbox"]
                confidence = pred["confidence"]
                if confidence < self.conf_thresh:
                    continue

                x1 = (x_center - width / 2.0) * w
                y1 = (y_center - height / 2.0) * h
                w_box = width * w
                h_box = height * h

                rect = plt.Rectangle((x1, y1), w_box, h_box,
                                     fill=False, color="red", linewidth=2, alpha=0.8)
                ax.add_patch(rect)
                ax.text(x1, y1 - 25, f"P:{confidence:.2f}", fontsize=9, color="red", weight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
                pred_count += 1

            legend_elements = [
                Line2D([0], [0], color="lime", lw=3, label=f"Ground Truth ({gt_count})"),
                Line2D([0], [0], color="red",  lw=2, label=f"Predictions ({pred_count})"),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=12)

            ax.set_title(f"Epoch {epoch}, Batch {batch_idx} | GT: {gt_count}, Pred: {pred_count}",
                         fontsize=14, weight="bold")
            ax.axis("off")

            filename = f"epoch_{epoch:03d}_batch_{batch_idx:04d}_gt{gt_count}_pred{pred_count}.png"
            save_path = self.vis_dir / filename
            fig.savefig(save_path, bbox_inches="tight", dpi=100)
            plt.close(fig)

            self.batch_count += 1
            return str(save_path)

        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return None
