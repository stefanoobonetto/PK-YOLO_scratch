import os
import time
import torch
import logging
import warnings
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import freeze_support
try:
    import cv2
    cv2.setNumThreads(0)
except Exception:
    pass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass


from complete_yolo_loss import YOLOLoss
try:
    from utils.utils import get_train_arg_parser
except Exception:
    from utils import get_train_arg_parser
try:
    from utils.early_stopping import EarlyStopping
except Exception:
    from early_stopping import EarlyStopping
from multimodal_pk_yolo import MultimodalPKYOLO
from brats_dataset import BraTSDataset, collate_fn
from torch.utils.data import WeightedRandomSampler

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Visualizer:
    """Training visualization"""
    def __init__(self, output_dir: str, save_interval: int = 100):
        self.vis_dir = Path(output_dir) / 'training_visualizations'
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
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
        """
        Decode model outputs to normalized YOLO-style [cx, cy, w, h] + confidence.
        Returns a list of dicts: {'bbox':[cx,cy,w,h], 'confidence':float}
        """

        import torch

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

            # probabilities
            cls_prob = torch.sigmoid(cls_score)[0, ..., 0]    # (3,H,W)
            obj_prob = torch.sigmoid(objectness)[0]           # (3,H,W)
            conf = (cls_prob * obj_prob)                      # (3,H,W)

            # grid and anchors (grid in (x,y) order)
            device = cls_score.device
            gy, gx = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing='ij'
            )  # (H,W)
            grid = torch.stack((gx, gy), dim=-1).float()      # (H,W,2)

            scale_anchors = torch.tensor(anchors[scale_idx], device=device, dtype=torch.float32) / stride  # (3,2)

            # decode xywh → normalized
            box = bbox_pred[0]                                # (3,H,W,4)
            xy = torch.sigmoid(box[..., 0:2]) * 2.0 - 0.5     # (3,H,W,2)
            wh = (torch.sigmoid(box[..., 2:4]) * 2.0) ** 2    # (3,H,W,2)
            wh = wh * scale_anchors[:, None, None, :]         # multiply by anchor per a

            # centers in pixels / normalized
            cx = (xy[..., 0] + grid[..., 0]) * stride / img_size  # (3,H,W)
            cy = (xy[..., 1] + grid[..., 1]) * stride / img_size  # (3,H,W)
            ww = (wh[..., 0] * stride) / img_size
            hh = (wh[..., 1] * stride) / img_size

            # filter by conf and by reasonable sizes
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

        # NMS in pixel space
        xyxy_pix = xyxy * float(img_size)
        # Simple, dependency-free NMS
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
            # extra guard
            if ww <= 0 or hh <= 0:
                continue
            out.append({'bbox': [cx, cy, ww, hh], 'confidence': float(scores[idx])})

        # sort by confidence and return (no hard cap to 5)
        return sorted(out, key=lambda d: d['confidence'], reverse=True)

    # Fix aggiuntivo per il caricamento delle immagini nel dataset
    def load_multimodal_image(self, slice_id):
        """Load all 4 modalities with better error handling"""
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
                    if img is not None and img.size > 0:  # Verifica che l'immagine sia valida
                        break
                    else:
                        img = None
            
            if img is None:
                logger.warning(f"Missing or corrupted {modality} for {slice_id}")
                img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                # Verifica che l'immagine non sia corrotta
                if img.std() < 1.0:  # Immagine troppo uniforme (possibile corruzione)
                    logger.warning(f"Potentially corrupted {modality} for {slice_id} (low std: {img.std()})")
                
                if img.shape[:2] != (self.img_size, self.img_size):
                    img = cv2.resize(img, (self.img_size, self.img_size))
            
            images.append(img)
        
        multimodal_img = np.stack(images, axis=-1)
        return multimodal_img

    def save_visualization(self, batch_idx: int, epoch: int, images: torch.Tensor, targets: dict, predictions, slice_ids: list):
        """Save visualization with GT and predictions"""
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
                conf_thresh=0.25,
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
            
            # Draw prediction boxes
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
            
            from matplotlib.lines import Line2D
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

class Trainer:    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
        self.model = MultimodalPKYOLO(
            num_classes=self.config.get('model.num_classes', 1),
            input_channels=self.config.get('model.input_channels', 4)
        ).float().to(self.device)
                
        self.criterion = YOLOLoss(
            model=self.model,
            num_classes=self.config.get('model.num_classes', 1),
            autobalance=True
        )
        
        optimizer_type = self.config.get('optimizer.type', 'AdamW')
        lr = self.config.get('training.learning_rate', 1e-3)
        weight_decay = self.config.get('training.weight_decay', 1e-4)
        
        if optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        scheduler_type = self.config.get('optimizer.lr_scheduler', 'CosineAnnealingLR')
        num_epochs = self.config.get('training.num_epochs', 100)
        
        if scheduler_type == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=10)
        elif scheduler_type == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        else:
            self.scheduler = None
            
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        self.output_dir = Path(self.config.get('logging.output_dir', 'outputs'))
        self.output_dir.mkdir(exist_ok=True)
        
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        if self.config.get('training.early_stopping', True):
            self.early_stopping = EarlyStopping(
                patience=self.config.get('training.patience', 20),
                min_delta=self.config.get('training.min_delta', 0.001)
            )
        else:
            self.early_stopping = None
        
        self.scaler = None
        if self.config.get('training.mixed_precision', False):
            try:
                self.scaler = torch.cuda.amp.GradScaler()
            except:
                self.scaler = None
        
        self.visualizer = Visualizer(
            output_dir=str(self.output_dir),
            save_interval=self.config.get('visualization.save_interval', 100)
        )
                
    def load_datasets(self):
        data_dir = self.config.get('data.data_dir', './data')
        img_size = self.config.get('model.img_size', 640)
        batch_size = self.config.get('training.batch_size', 8)
        num_workers = self.config.get('data.num_workers', 4)
        # Force 0 workers on Windows to avoid spawn-related hangs
        if os.name == 'nt':
            num_workers = 0
        pin_memory = self.config.get('data.pin_memory', True) and (self.device.type == 'cuda')
        
        train_dir = Path(data_dir) / 'train'
        val_dir = Path(data_dir) / 'val'
        test_dir = Path(data_dir) / 'test'
        
        if not train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        
        if val_dir.exists():
            val_split_name = 'val'
        elif test_dir.exists():
            val_split_name = 'test'
        else:
            raise FileNotFoundError(f"Neither val nor test directory found in {data_dir}")
        
        logger.info(f"Using {val_split_name} as validation set")
        
        self.train_dataset = BraTSDataset(
            data_dir, split='train', img_size=img_size,
            augment=self.config.get('augmentation.enabled', True)
        )
        
        def is_positive_label(txt_path):
            if not os.path.exists(txt_path):
                return 0.0
            try:
                return 1.0 if os.path.getsize(txt_path) > 0 else 0.0  # file non vuoto => almeno 1 box
            except:
                return 0.0

        # Costruisci la lista dei label path dal dataset (adatta se il tuo dataset espone un attributo diverso)
        # Build label paths using slice_ids (one label per slice)
        label_paths = [str(Path(train_dir) / "labels" / f"{sid}.txt") for sid in self.train_dataset.slice_ids]

        weights = []
        for p in label_paths:
            pos = is_positive_label(p)
            # dai più peso alle positive (es. 3x). Tieni 1x per le negative
            weights.append(3.0 if pos > 0 else 1.0)

        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0)
        )
        
        self.val_dataset = BraTSDataset(
            data_dir, split=val_split_name, img_size=img_size, augment=False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory
        )
        
        logger.info(f"Loaded {len(self.train_dataset)} training samples")
        logger.info(f"Loaded {len(self.val_dataset)} validation samples")

        # Quick sanity check: attempt to load a single sample now so failures happen early & clearly
        try:
            logger.info("Sanity-check: loading one training sample...")
            _sample = self.train_dataset[0]
            _img = _sample['image']
            if isinstance(_img, torch.Tensor):
                logger.info(f"Sample image tensor shape: {_img.shape}; bboxes: {len(_sample['bboxes'])}")
            else:
                logger.info("Sample image loaded (non-tensor).")
        except Exception as e:
            logger.error(f"Sanity-check failed while loading a training sample: {e}")
            raise
        
        if len(self.train_dataset) == 0:
            raise ValueError("Training dataset is empty")
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_components = {'box_loss': 0.0, 'obj_loss': 0.0, 'cls_loss': 0.0}
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                images = batch['images'].to(self.device, dtype=torch.float32, non_blocking=True)
                targets = {
                    'bboxes': batch['bboxes'].to(self.device, dtype=torch.float32, non_blocking=True),
                    'labels': batch['labels'].to(self.device, non_blocking=True),
                    'images': images
                }
                
                self.optimizer.zero_grad()
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(images)
                        loss, loss_components = self.criterion(predictions, targets)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    predictions = self.model(images)
                    loss, loss_components = self.criterion(predictions, targets)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.optimizer.step()
                
                total_loss += loss.item()
                total_components['box_loss'] += loss_components[0].item()
                total_components['obj_loss'] += loss_components[1].item()
                total_components['cls_loss'] += loss_components[2].item()
                
                if self.visualizer.should_save(batch_idx):
                    slice_ids = batch.get('slice_ids', [f'batch_{batch_idx}'])
                    self.visualizer.save_visualization(
                        batch_idx, self.current_epoch, images, targets, predictions, slice_ids
                    )
                
                if batch_idx % 10 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Box': f'{loss_components[0].item():.4f}',
                        'Obj': f'{loss_components[1].item():.4f}',
                        'Cls': f'{loss_components[2].item():.4f}',
                        'LR': f'{current_lr:.6f}'
                    })
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"CUDA OOM at batch {batch_idx}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in total_components.items()}
        
        return avg_loss, avg_components
    
    def validate_epoch(self):
        self.model.eval()
        
        total_loss = 0.0
        total_components = {'box_loss': 0.0, 'obj_loss': 0.0, 'cls_loss': 0.0}
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                try:
                    images = batch['images'].to(self.device, dtype=torch.float32, non_blocking=True)
                    targets = {
                        'bboxes': batch['bboxes'].to(self.device, dtype=torch.float32, non_blocking=True),
                        'labels': batch['labels'].to(self.device, non_blocking=True),
                        'images': images
                    }
                    
                    predictions = self.model(images)
                    loss, loss_components = self.criterion(predictions, targets)
                    
                    total_loss += loss.item()
                    total_components['box_loss'] += loss_components[0].item()
                    total_components['obj_loss'] += loss_components[1].item()
                    total_components['cls_loss'] += loss_components[2].item()
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in total_components.items()}
        
        return avg_loss, avg_components
    
    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
        }
        
        checkpoint_path = self.output_dir / 'checkpoints' / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.output_dir / 'checkpoints' / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def train(self):
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        num_epochs = self.config.get('training.num_epochs', 100)
        save_interval = self.config.get('logging.save_interval', 25)
        
        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                train_loss, train_components = self.train_epoch()
                self.train_losses.append(train_loss)
                
                val_loss, val_components = self.validate_epoch()
                self.val_losses.append(val_loss)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)
                
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif self.scheduler:
                    self.scheduler.step()
                
                epoch_time = time.time() - epoch_start_time
                
                logger.info(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {epoch_time:.1f}s"
                )
                
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint('best_model.pth', is_best=True)
                    logger.info(f"New best loss: {self.best_loss:.4f}")
                
                if self.early_stopping:
                    if self.early_stopping(val_loss, self.model):
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
                
                if epoch % save_interval == 0 and epoch > 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self.save_checkpoint('final_model.pth')
            logger.info(f"Training completed. Saved {self.visualizer.batch_count} visualizations")

def create_default_config(args=None):
    config = {
        'model': {
            'num_classes': 1,
            'input_channels': 4,
            'img_size': 640,
        },
        'training': {
            'batch_size': 4,
            'num_epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'mixed_precision': True,
            'early_stopping': True,
            'patience': 20,
            'min_delta': 0.001
        },
        'data': {
            'data_dir': './data',
            'num_workers': 4,
            'pin_memory': True
        },
        'augmentation': {
            'enabled': True,
        },
        'optimizer': {
            'type': 'AdamW',
            'lr_scheduler': 'CosineAnnealingLR'
        },
        'logging': {
            'output_dir': 'outputs',
            'save_interval': 25
        },
        'visualization': {
            'save_interval': 500,
        }
    }
    
    if args:
        if args.data_dir:
            config['data']['data_dir'] = args.data_dir
        if args.output_dir:
            config['logging']['output_dir'] = args.output_dir
        if args.batch_size:
            config['training']['batch_size'] = args.batch_size
        if args.epochs:
            config['training']['num_epochs'] = args.epochs
        if args.lr:
            config['training']['learning_rate'] = args.lr
        if args.img_size:
            config['model']['img_size'] = args.img_size
        if args.workers:
            config['data']['num_workers'] = args.workers
        if args.mixed_precision:
            config['training']['mixed_precision'] = True
    
    return config

class SimpleConfig:
    def __init__(self, config_dict=None):
        self.config = config_dict or create_default_config()
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

def main():
    parser = get_train_arg_parser()
    args = parser.parse_args()
    
    config_dict = create_default_config(args)
    config = SimpleConfig(config_dict)
    
    logger.info("MULTIMODAL PK-YOLO TRAINING")
    logger.info(f"Data directory: {config.get('data.data_dir')}")
    logger.info(f"Output directory: {config.get('logging.output_dir')}")
    logger.info(f"Batch size: {config.get('training.batch_size')}")
    logger.info(f"Epochs: {config.get('training.num_epochs')}")
    
    try:
        trainer = Trainer(config)
        trainer.load_datasets()
        trainer.train()
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    freeze_support()
    main()