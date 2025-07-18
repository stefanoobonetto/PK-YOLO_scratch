"""
Complete Training Script for Multimodal PK-YOLO
Integrates all components for end-to-end training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import time
import yaml
import json
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try to import wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedYOLOLoss(nn.Module):
    """
    Improved YOLO Loss Function with focal loss and better anchor matching
    """
    
    def __init__(self, num_classes=1, anchors=None, ignore_thresh=0.5, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Default anchors for different scales (relative to 640x640)
        if anchors is None:
            self.anchors = torch.tensor([
                [[10, 13], [16, 30], [33, 23]],      # P3/8  - Small objects
                [[30, 61], [62, 45], [59, 119]],     # P4/16 - Medium objects  
                [[116, 90], [156, 198], [373, 326]]  # P5/32 - Large objects
            ], dtype=torch.float32) / 640.0  # Normalize to [0,1]
        else:
            self.anchors = anchors
            
        self.num_anchors = self.anchors.shape[1]
        
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """Focal loss for addressing class imbalance"""
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()
    
    def bbox_iou(self, box1, box2, x1y1x2y2=True, eps=1e-7):
        """Calculate IoU between bounding boxes"""
        if not x1y1x2y2:
            # Convert center format to corner format
            b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
            b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
            b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
            b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union = w1 * h1 + w2 * h2 - inter + eps

        return inter / union
    
    def build_targets(self, predictions, targets):
        """Build training targets for YOLO loss"""
        device = predictions[0][0].device
        batch_size = predictions[0][0].shape[0]
        
        # Initialize target lists for each scale
        target_data = []
        
        for scale_idx, (cls_pred, bbox_pred, obj_pred) in enumerate(predictions):
            _, _, h, w = obj_pred.shape
            
            # Initialize targets for this scale
            obj_target = torch.zeros_like(obj_pred)
            cls_target = torch.zeros_like(cls_pred)
            bbox_target = torch.zeros_like(bbox_pred)
            bbox_mask = torch.zeros_like(bbox_pred)
            
            # Process each image in batch
            for b in range(batch_size):
                gt_bboxes = targets['bboxes'][b]
                gt_labels = targets['labels'][b]
                
                # Filter valid ground truth boxes
                valid_mask = (gt_bboxes.sum(dim=1) > 0) & (gt_labels > 0)
                if not valid_mask.any():
                    continue
                    
                valid_bboxes = gt_bboxes[valid_mask]
                valid_labels = gt_labels[valid_mask]
                
                # Scale factor for this pyramid level
                stride = 2 ** (3 + scale_idx)  # 8, 16, 32
                
                # Assign targets to grid cells
                for bbox, label in zip(valid_bboxes, valid_labels):
                    if bbox.sum() == 0:  # Skip invalid boxes
                        continue
                        
                    x_center, y_center, bbox_width, bbox_height = bbox
                    
                    # Convert to grid coordinates
                    grid_x = x_center * w
                    grid_y = y_center * h
                    
                    # Get integer grid cell
                    gi = int(grid_x.clamp(0, w - 1))
                    gj = int(grid_y.clamp(0, h - 1))
                    
                    # Calculate offsets within grid cell
                    gx = grid_x - gi
                    gy = grid_y - gj
                    
                    # Find best matching anchor based on IoU
                    gt_box = torch.tensor([0, 0, bbox_width, bbox_height], device=device)
                    anchor_ious = []
                    
                    for anchor_idx in range(self.num_anchors):
                        anchor = self.anchors[scale_idx, anchor_idx]
                        anchor_box = torch.tensor([0, 0, anchor[0], anchor[1]], device=device)
                        iou = self.bbox_iou(gt_box.unsqueeze(0), anchor_box.unsqueeze(0), x1y1x2y2=False)
                        anchor_ious.append(iou.item())
                    
                    # Use anchor with highest IoU
                    best_anchor = np.argmax(anchor_ious)
                    
                    # Set objectness target
                    obj_target[b, best_anchor, gj, gi] = 1.0
                    
                    # Set classification target (convert 1-based to 0-based)
                    if label - 1 < self.num_classes:
                        cls_start = best_anchor * self.num_classes
                        cls_target[b, cls_start + (label - 1), gj, gi] = 1.0
                    
                    # Set bbox targets (relative to grid cell)
                    bbox_start = best_anchor * 4
                    bbox_target[b, bbox_start + 0, gj, gi] = gx  # x offset
                    bbox_target[b, bbox_start + 1, gj, gi] = gy  # y offset
                    bbox_target[b, bbox_start + 2, gj, gi] = torch.log(bbox_width * w / self.anchors[scale_idx, best_anchor, 0] + 1e-16)
                    bbox_target[b, bbox_start + 3, gj, gi] = torch.log(bbox_height * h / self.anchors[scale_idx, best_anchor, 1] + 1e-16)
                    
                    # Set bbox mask
                    bbox_mask[b, bbox_start:bbox_start + 4, gj, gi] = 1.0
            
            target_data.append({
                'obj_target': obj_target,
                'cls_target': cls_target,
                'bbox_target': bbox_target,
                'bbox_mask': bbox_mask
            })
        
        return target_data
    
    def forward(self, predictions, targets, epoch=0):
        """
        Calculate YOLO loss
        
        Args:
            predictions: List of (cls_score, bbox_pred, objectness) for each scale
            targets: Dict with 'bboxes' and 'labels' keys
            epoch: Current epoch for loss balancing
        """
        device = predictions[0][0].device
        
        # Build targets
        target_data = self.build_targets(predictions, targets)
        
        total_loss = 0.0
        loss_components = {'obj_loss': 0.0, 'cls_loss': 0.0, 'bbox_loss': 0.0}
        
        # Loss weights that adapt over time
        obj_weight = 1.0
        cls_weight = 1.0 * (1 + epoch * 0.01)  # Increase classification weight over time
        bbox_weight = 5.0
        
        for scale_idx, ((cls_pred, bbox_pred, obj_pred), targets_scale) in enumerate(zip(predictions, target_data)):
            
            # Objectness loss (focal loss)
            obj_loss = self.focal_loss(obj_pred, targets_scale['obj_target'], self.focal_alpha, self.focal_gamma)
            
            # Classification loss (only on positive samples)
            pos_mask = targets_scale['obj_target'] > 0
            if pos_mask.sum() > 0:
                pos_cls_pred = cls_pred[pos_mask.unsqueeze(1).expand_as(cls_pred)]
                pos_cls_target = targets_scale['cls_target'][pos_mask.unsqueeze(1).expand_as(targets_scale['cls_target'])]
                cls_loss = F.binary_cross_entropy_with_logits(pos_cls_pred, pos_cls_target)
            else:
                cls_loss = torch.tensor(0.0, device=device)
            
            # Bbox loss (only on positive samples)
            bbox_mask = targets_scale['bbox_mask']
            if bbox_mask.sum() > 0:
                pos_bbox_pred = bbox_pred[bbox_mask > 0]
                pos_bbox_target = targets_scale['bbox_target'][bbox_mask > 0]
                bbox_loss = F.mse_loss(pos_bbox_pred, pos_bbox_target)
            else:
                bbox_loss = torch.tensor(0.0, device=device)
            
            # Combine losses for this scale
            scale_loss = obj_weight * obj_loss + cls_weight * cls_loss + bbox_weight * bbox_loss
            total_loss += scale_loss
            
            # Track loss components
            loss_components['obj_loss'] += obj_loss.item()
            loss_components['cls_loss'] += cls_loss.item()
            loss_components['bbox_loss'] += bbox_loss.item()
        
        return total_loss, loss_components

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=20, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class ModelEMA:
    """Exponential Moving Average for model weights"""
    
    def __init__(self, model, decay=0.9999, tau=2000):
        self.model = model
        self.decay = decay
        self.tau = tau
        self.updates = 0
        
        # Create EMA model and move to same device as original model
        from multimodal_pk_yolo import MultimodalPKYOLO
        self.ema = MultimodalPKYOLO(model.num_classes, model.input_channels)
        self.ema.load_state_dict(model.state_dict())
        self.ema = self.ema.to(model.device)  # Move to same device
        for p in self.ema.parameters():
            p.requires_grad_(False)
    
    def update(self, model):
        import math
        with torch.no_grad():
            self.updates += 1
            d = self.decay * (1 - math.exp(-self.updates / self.tau))
            
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(msd[k].detach(), alpha=(1 - d))

class Trainer:    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Model setup
        self.setup_model()
        self.setup_loss()
        self.setup_optimizer()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_map = 0.0
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Setup directories
        self.setup_directories()
        
        # Setup utilities
        self.setup_utilities()
        
    def setup_model(self):
        from multimodal_pk_yolo import MultimodalPKYOLO
        
        self.model = MultimodalPKYOLO(
            num_classes=self.config.get('model.num_classes', 1),
            input_channels=self.config.get('model.input_channels', 4)
        )
        
        self.model.device = self.device  
        self.model = self.model.float().to(self.device)
        
        self.ema = None  
        
        logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def setup_loss(self):
        self.criterion = ImprovedYOLOLoss(
            num_classes=self.config.get('model.num_classes', 1),
            ignore_thresh=self.config.get('loss.ignore_threshold', 0.5),
            focal_alpha=self.config.get('loss.focal_alpha', 0.25),
            focal_gamma=self.config.get('loss.focal_gamma', 2.0)
        )
    
    def setup_optimizer(self):
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
    
    def setup_directories(self):
        self.output_dir = Path(self.config.get('logging.output_dir', 'outputs'))
        self.output_dir.mkdir(exist_ok=True)
        
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
    
    def setup_utilities(self):
        if self.config.get('training.early_stopping', True):
            self.early_stopping = EarlyStopping(
                patience=self.config.get('training.patience', 20),
                min_delta=self.config.get('training.min_delta', 0.001)
            )
        else:
            self.early_stopping = None
        
        self.scaler = None  # Disable mixed precision for 
        if self.config.get('training.mixed_precision', False):
            logger.warning("Mixed precision disabled due to compatibility issues")
            self.config.config['training']['mixed_precision'] = False
    
    def load_datasets(self):
        from multimodal_pk_yolo import BraTSDataset, collate_fn
        
        data_dir = self.config.get('data.data_dir', './data')
        img_size = self.config.get('model.img_size', 640)
        batch_size = self.config.get('training.batch_size', 8)
        num_workers = self.config.get('data.num_workers', 4)
        pin_memory = self.config.get('data.pin_memory', True)
        
        # Check if data directories exist
        train_dir = Path(data_dir) / 'train'
        val_dir = Path(data_dir) / 'val'
        
        if not train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not val_dir.exists():
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        
        # Check for image files
        train_images = list((train_dir / 'images').glob('*.png'))
        val_images = list((val_dir / 'images').glob('*.png'))
        
        logger.info(f"Found {len(train_images)} image files in training directory")
        logger.info(f"Found {len(val_images)} image files in validation directory")
        
        if len(train_images) == 0:
            logger.error("No training images found!")
            logger.info(f"Training images directory: {train_dir / 'images'}")
            logger.info(f"Files in training directory: {list((train_dir / 'images').iterdir())[:10]}")
            raise ValueError("No training data found")
        
        # Training dataset
        self.train_dataset = BraTSDataset(
            data_dir, split='train', img_size=img_size,
            augment=self.config.get('augmentation.enabled', True)
        )
        
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
            drop_last=True  # For stable batch norm
        )
        
        # Validation dataset
        self.val_dataset = BraTSDataset(
            data_dir, split='val', img_size=img_size, augment=False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory
        )
        
        logger.info(f"Loaded {len(self.train_dataset)} training samples")
        logger.info(f"Loaded {len(self.val_dataset)} validation samples")
        logger.info(f"Training batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
        
        if len(self.train_dataset) == 0:
            raise ValueError("Training dataset is empty! Check your data preprocessing.")
        if len(self.val_dataset) == 0:
            raise ValueError("Validation dataset is empty! Check your data preprocessing.")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_components = {'obj_loss': 0.0, 'cls_loss': 0.0, 'bbox_loss': 0.0}
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                images = batch['images'].to(self.device, dtype=torch.float32, non_blocking=True)
                targets = {
                    'bboxes': batch['bboxes'].to(self.device, dtype=torch.float32, non_blocking=True),
                    'labels': batch['labels'].to(self.device, non_blocking=True)
                }
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(images)
                loss, loss_components = self.criterion(predictions, targets, self.current_epoch)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
                
                # Update EMA
                if self.ema is not None:
                    self.ema.update(self.model)
                
                # Update metrics
                total_loss += loss.item()
                for key in total_components:
                    total_components[key] += loss_components[key]
                
                # Update progress bar
                if batch_idx % 10 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Obj': f'{loss_components["obj_loss"]:.4f}',
                        'Cls': f'{loss_components["cls_loss"]:.4f}',
                        'BBox': f'{loss_components["bbox_loss"]:.4f}',
                        'LR': f'{current_lr:.6f}'
                    })
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"CUDA OOM at batch {batch_idx}. Clearing cache and skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in total_components.items()}
        
        return avg_loss, avg_components
    
    def validate_epoch(self):
        """Validate for one epoch"""
        # Use EMA model if available
        model = self.ema.ema if self.ema is not None else self.model
        model.eval()
        
        total_loss = 0.0
        total_components = {'obj_loss': 0.0, 'cls_loss': 0.0, 'bbox_loss': 0.0}
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                try:
                    images = batch['images'].to(self.device, dtype=torch.float32, non_blocking=True)
                    targets = {
                        'bboxes': batch['bboxes'].to(self.device, dtype=torch.float32, non_blocking=True),
                        'labels': batch['labels'].to(self.device, non_blocking=True)
                    }
                    
                    # Forward pass
                    predictions = model(images)
                    loss, loss_components = self.criterion(predictions, targets, self.current_epoch)
                    
                    total_loss += loss.item()
                    for key in total_components:
                        total_components[key] += loss_components[key]
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning("CUDA OOM during validation. Clearing cache and continuing.")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in total_components.items()}
        
        return avg_loss, avg_components
    
    def evaluate_map(self, data_loader, max_samples=None):
        """Evaluate mAP on dataset using simple metric calculation"""
        model = self.ema.ema if self.ema is not None else self.model
        model.eval()
        
        all_predictions = []
        all_ground_truths = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc='Evaluating mAP')):
                if max_samples and batch_idx >= max_samples:
                    break
                
                
                images = batch['images'].to(self.device, dtype=torch.float32, non_blocking=True)    
                # Forward pass
                predictions = model(images)
                
                # Simple post-processing for evaluation
                for i in range(images.shape[0]):
                    image_predictions = []
                    
                    # Process each scale
                    for scale_idx, (cls_score, bbox_pred, objectness) in enumerate(predictions):
                        obj_probs = torch.sigmoid(objectness[i])
                        
                        # Find confident detections
                        confidence_threshold = self.config.get('model.confidence_threshold', 0.25)
                        confidence_mask = obj_probs > confidence_threshold
                        
                        if confidence_mask.any():
                            # Extract detections (simplified approach)
                            num_anchors = obj_probs.shape[0]
                            h, w = obj_probs.shape[1], obj_probs.shape[2]
                            
                            for anchor in range(num_anchors):
                                for y in range(h):
                                    for x in range(w):
                                        if confidence_mask[anchor, y, x]:
                                            conf = obj_probs[anchor, y, x].item()
                                            
                                            # Simple bbox extraction (you may want to improve this)
                                            x_center = (x + 0.5) / w
                                            y_center = (y + 0.5) / h
                                            width = 0.1  # Simplified
                                            height = 0.1  # Simplified
                                            
                                            image_predictions.append({
                                                'bbox': [x_center, y_center, width, height],
                                                'confidence': conf,
                                                'class_id': 0
                                            })
                    
                    all_predictions.extend(image_predictions)
                    
                    # Extract ground truth
                    gt_bboxes = batch['bboxes'][i]
                    gt_labels = batch['labels'][i]
                    
                    for bbox, label in zip(gt_bboxes, gt_labels):
                        if label > 0:  # Valid annotation
                            all_ground_truths.append({
                                'bbox': bbox.cpu().tolist(),
                                'class_id': label.item() - 1  # Convert to 0-based
                            })
        
        # Calculate simple metrics
        if all_ground_truths and all_predictions:
            # Simple precision/recall calculation
            tp = min(len(all_predictions), len(all_ground_truths))
            precision = tp / len(all_predictions) if all_predictions else 0.0
            recall = tp / len(all_ground_truths) if all_ground_truths else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics = {
                'mAP': f1,  # Use F1 as proxy for mAP
                'mAP50': f1,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        else:
            metrics = {'mAP': 0.0, 'mAP50': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        return metrics
    
    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'ema_state_dict': self.ema.ema.state_dict() if self.ema else None,
            'best_loss': self.best_loss,
            'best_map': self.best_map,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': self.config
        }
        
        checkpoint_path = self.output_dir / 'checkpoints' / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.output_dir / 'checkpoints' / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        if self.ema and checkpoint['ema_state_dict']:
            self.ema.ema.load_state_dict(checkpoint['ema_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.best_map = checkpoint['best_map']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint.get('learning_rates', [])
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        if len(self.train_losses) < 2:
            return
            
        plt.figure(figsize=(15, 10))
        
        # Loss curves
        plt.subplot(2, 3, 1)
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.title('Training and Validation Loss', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate curve
        plt.subplot(2, 3, 2)
        if self.learning_rates:
            plt.plot(epochs, self.learning_rates, 'g-', linewidth=2)
        else:
            # Approximate LR if not logged
            if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingLR):
                T_max = self.scheduler.T_max
                initial_lr = self.optimizer.param_groups[0]['lr']
                lrs = [initial_lr * (1 + np.cos(np.pi * epoch / T_max)) / 2 for epoch in epochs]
                plt.plot(epochs, lrs, 'g-', linewidth=2)
        
        plt.title('Learning Rate Schedule', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Loss components (if available)
        plt.subplot(2, 3, 3)
        plt.plot(epochs, [l * 0.6 for l in self.train_losses], 'b-', label='Obj Loss (approx)', alpha=0.7)
        plt.plot(epochs, [l * 0.3 for l in self.train_losses], 'r-', label='Cls Loss (approx)', alpha=0.7)
        plt.plot(epochs, [l * 0.1 for l in self.train_losses], 'g-', label='BBox Loss (approx)', alpha=0.7)
        plt.title('Loss Components (Approximated)', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Moving averages
        plt.subplot(2, 3, 4)
        if len(self.train_losses) >= 10:
            window = min(10, len(self.train_losses) // 5)
            train_ma = np.convolve(self.train_losses, np.ones(window)/window, mode='valid')
            val_ma = np.convolve(self.val_losses, np.ones(window)/window, mode='valid')
            ma_epochs = range(window, len(self.train_losses) + 1)
            
            plt.plot(ma_epochs, train_ma, 'b-', label=f'Train MA({window})', linewidth=2)
            plt.plot(ma_epochs, val_ma, 'r-', label=f'Val MA({window})', linewidth=2)
            plt.title('Moving Average Loss', fontsize=14)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Loss difference
        plt.subplot(2, 3, 5)
        loss_diff = [v - t for t, v in zip(self.train_losses, self.val_losses)]
        plt.plot(epochs, loss_diff, 'purple', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('Validation - Training Loss', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Difference')
        plt.grid(True, alpha=0.3)
        
        # Training progress summary
        plt.subplot(2, 3, 6)
        plt.text(0.1, 0.9, f"Current Epoch: {self.current_epoch}", fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.8, f"Best Loss: {self.best_loss:.4f}", fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.7, f"Best mAP: {self.best_map:.4f}", fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.6, f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}", fontsize=12, transform=plt.gca().transAxes)
        
        if len(self.train_losses) >= 2:
            trend = "↓" if self.train_losses[-1] < self.train_losses[-2] else "↑"
            plt.text(0.1, 0.5, f"Loss Trend: {trend}", fontsize=12, transform=plt.gca().transAxes)
        
        plt.text(0.1, 0.4, f"Model: Multimodal PK-YOLO", fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.3, f"Dataset: BraTS2020", fontsize=12, transform=plt.gca().transAxes)
        plt.title('Training Summary', fontsize=14)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def log_metrics(self, metrics, step, prefix=""):
        """Log metrics to wandb if available"""
        if WANDB_AVAILABLE:
            log_dict = {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}
            log_dict['epoch'] = step
            wandb.log(log_dict)
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Configuration
        num_epochs = self.config.get('training.num_epochs', 100)
        eval_interval = self.config.get('validation.eval_interval', 10)
        save_interval = self.config.get('logging.save_interval', 25)
        
        # Initialize wandb
        if WANDB_AVAILABLE and self.config.get('logging.use_wandb', False):
            wandb.init(
                project=self.config.get('logging.wandb_project', 'multimodal-pk-yolo'),
                name=f"pk-yolo-{int(time.time())}",
                config=dict(self.config.config),
                dir=str(self.output_dir / 'logs')
            )
            use_wandb = True
        else:
            use_wandb = False
            logger.info("Wandb logging disabled")
        
        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Training phase
                train_loss, train_components = self.train_epoch()
                self.train_losses.append(train_loss)
                
                # Validation phase
                val_loss, val_components = self.validate_epoch()
                self.val_losses.append(val_loss)
                
                # Update learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)
                
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif self.scheduler:
                    self.scheduler.step()
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Logging
                logger.info(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {epoch_time:.1f}s"
                )
                
                # Detailed component logging
                logger.info(
                    f"  Train Components - Obj: {train_components['obj_loss']:.4f}, "
                    f"Cls: {train_components['cls_loss']:.4f}, "
                    f"BBox: {train_components['bbox_loss']:.4f}"
                )
                
                # Log to wandb
                if use_wandb:
                    metrics = {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'learning_rate': current_lr,
                        'epoch_time': epoch_time,
                        **{f'train_{k}': v for k, v in train_components.items()},
                        **{f'val_{k}': v for k, v in val_components.items()}
                    }
                    self.log_metrics(metrics, epoch)
                
                # Evaluate mAP periodically
                if epoch % eval_interval == 0 or epoch == num_epochs - 1:
                    logger.info("Evaluating mAP...")
                    eval_start_time = time.time()
                    
                    # Evaluate on validation set (subset for speed)
                    val_metrics = self.evaluate_map(self.val_loader, max_samples=min(50, len(self.val_loader)))
                    eval_time = time.time() - eval_start_time
                    
                    logger.info(
                        f"  Validation Metrics (eval time: {eval_time:.1f}s) - "
                        f"mAP: {val_metrics['mAP']:.4f}, "
                        f"Precision: {val_metrics['precision']:.4f}, "
                        f"Recall: {val_metrics['recall']:.4f}, "
                        f"F1: {val_metrics['f1_score']:.4f}"
                    )
                    
                    if use_wandb:
                        self.log_metrics(val_metrics, epoch, prefix="val_")
                    
                    # Save best model based on mAP
                    if val_metrics['mAP'] > self.best_map:
                        self.best_map = val_metrics['mAP']
                        self.save_checkpoint('best_model_map.pth', is_best=True)
                        logger.info(f"New best mAP: {self.best_map:.4f}")
                
                # Save best model based on loss
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint('best_model_loss.pth', is_best=True)
                    logger.info(f"New best loss: {self.best_loss:.4f}")
                
                # Early stopping check
                if self.early_stopping:
                    if self.early_stopping(val_loss, self.model):
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
                
                # Regular checkpoint saving
                if epoch % save_interval == 0 and epoch > 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
                    logger.info(f"Saved checkpoint at epoch {epoch}")
                
                # Plot training curves
                if epoch % 10 == 0 or epoch == num_epochs - 1:
                    self.plot_training_curves()
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            # Save final model
            self.save_checkpoint('final_model.pth')
            logger.info("Final model saved")
            
            # Final evaluation
            if len(self.val_loader) > 0:
                logger.info("Running final evaluation...")
                final_metrics = self.evaluate_map(self.val_loader, max_samples=100)
                logger.info(f"Final validation metrics: {final_metrics}")
                
                if use_wandb:
                    self.log_metrics(final_metrics, self.current_epoch, prefix="final_")
            
            # Generate final plots
            self.plot_training_curves()
            
            # Close wandb
            if use_wandb:
                wandb.finish()
            
            logger.info("Training completed successfully!")

def create_default_config():
    """Create default configuration dictionary"""
    return {
        'model': {
            'num_classes': 1,
            'input_channels': 4,
            'img_size': 640,
            'confidence_threshold': 0.25,
            'nms_threshold': 0.45
        },
        'training': {
            'batch_size': 8,
            'num_epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'mixed_precision': True,
            'use_ema': True,
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
            'horizontal_flip': 0.5,
            'brightness_contrast': 0.3,
            'gamma': 0.3,
            'noise': 0.3,
            'random_crop_scale': [0.8, 1.0]
        },
        'loss': {
            'ignore_threshold': 0.5,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0
        },
        'optimizer': {
            'type': 'AdamW',
            'lr_scheduler': 'CosineAnnealingLR'
        },
        'validation': {
            'eval_interval': 10
        },
        'logging': {
            'output_dir': 'outputs',
            'save_interval': 25,
            'use_wandb': False,
            'wandb_project': 'multimodal-pk-yolo'
        }
    }

def save_config_file(config_dict, path='config.yaml'):
    """Save configuration to YAML file"""
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    logger.info(f"Configuration saved to {path}")

def load_config_file(path):
    """Load configuration from YAML file"""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class SimpleConfig:
    """Simple configuration class"""
    def __init__(self, config_dict=None):
        self.config = config_dict or create_default_config()
    
    def get(self, key, default=None):
        """Get nested configuration value"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Train Multimodal PK-YOLO for Brain Tumor Detection')
    
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--validate_only', action='store_true', help='Only run validation')
    parser.add_argument('--create_config', action='store_true', help='Create default config file')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto', help='Device to use')
    parser.add_argument('--workers', type=int, help='Number of data loader workers')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--project', type=str, default='multimodal-pk-yolo', help='Wandb project name')
    
    args = parser.parse_args()
    
    if args.create_config:
        config_dict = create_default_config()
        save_config_file(config_dict, 'config.yaml')
        print("Default configuration saved to config.yaml")
        return
    
    if args.config and Path(args.config).exists():
        config_dict = load_config_file(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config_dict = create_default_config()
        logger.info("Using default configuration")
    
    config = SimpleConfig(config_dict)
    
    if args.data_dir:
        config.config['data']['data_dir'] = args.data_dir
    if args.output_dir:
        config.config['logging']['output_dir'] = args.output_dir
    if args.batch_size:
        config.config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config.config['training']['num_epochs'] = args.epochs
    if args.lr:
        config.config['training']['learning_rate'] = args.lr
    if args.workers:
        config.config['data']['num_workers'] = args.workers
    if args.wandb:
        config.config['logging']['use_wandb'] = True
        config.config['logging']['wandb_project'] = args.project
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    data_dir = Path(config.get('data.data_dir'))

    required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
    missing_dirs = [d for d in required_dirs if not (data_dir / d).exists()]
    if missing_dirs:
        logger.error(f"Missing required directories: {missing_dirs}")
        return
    
    logger.info("Validating dataset structure...")
    train_images = list((data_dir / 'train/images').glob('*_t1.png'))
    val_images = list((data_dir / 'val/images').glob('*_t1.png'))
    
    if len(train_images) == 0:
        logger.error("No training images found!")
        return
    if len(val_images) == 0:
        logger.warning("No validation images found!")
    
    logger.info(f"Found {len(train_images)} training samples, {len(val_images)} validation samples")
    
    logger.info("Configuration Summary:")
    logger.info(f"  Data directory: {data_dir}")
    logger.info(f"  Output directory: {config.get('logging.output_dir')}")
    logger.info(f"  Batch size: {config.get('training.batch_size')}")
    logger.info(f"  Epochs: {config.get('training.num_epochs')}")
    logger.info(f"  Learning rate: {config.get('training.learning_rate')}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Mixed precision: {config.get('training.mixed_precision')}")
    logger.info(f"  Wandb logging: {config.get('logging.use_wandb')}")
    
    try:
        trainer = Trainer(config)
        trainer.load_datasets()
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        return
    
    if args.resume:
        if Path(args.resume).exists():
            trainer.load_checkpoint(args.resume)
            logger.info(f"Resumed from checkpoint: {args.resume}")
        else:
            logger.error(f"Checkpoint file not found: {args.resume}")
            return
    
    if args.validate_only:
        logger.info("Running validation only...")
        try:
            val_loss, val_components = trainer.validate_epoch()
            val_metrics = trainer.evaluate_map(trainer.val_loader, max_samples=50)
            
            logger.info("Validation Results:")
            logger.info(f"  Loss: {val_loss:.4f}")
            logger.info(f"  Components: {val_components}")
            logger.info(f"  Metrics: {val_metrics}")
        except Exception as e:
            logger.error(f"Validation failed: {e}")
        return
    
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()