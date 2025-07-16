"""
Multimodal PK-YOLO Implementation for Brain Tumor Detection
4-Channel Backbone Setup with RepViT + SparK + YOLOv9
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

import os
import cv2
import numpy as np
import glob
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import math
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DWConv(nn.Module):
    """Depth-wise convolution"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        return self.dwconv(x)

class RepViTBlock(nn.Module):
    """RepViT Block with transformer-like modules"""
    def __init__(self, inp, oup, stride=1, expand_ratio=4):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        
        # Depth-wise separable convolution
        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        # Depth-wise convolution
        self.dwconv = DWConv(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        # Point-wise convolution
        self.conv2 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)
        
        # Attention mechanism
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(oup, oup // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(oup // 16, oup, 1),
            nn.Sigmoid()
        )
        
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        if self.identity:
            identity = x
            
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.dwconv(out)))
        out = self.bn3(self.conv2(out))
        
        # Apply attention
        out = out * self.se(out)
        
        if self.identity:
            out += identity
            
        return out

class SparKRepViTBackbone(nn.Module):
    """
    SparK-pretrained RepViT backbone adapted for 4-channel multimodal input
    """
    def __init__(self, input_channels=4, width_mult=1.0):
        super().__init__()
        
        # Layer configurations: [channels, num_blocks, stride]
        self.cfgs = [
            [16, 1, 1],
            [32, 2, 2], 
            [64, 4, 2],
            [128, 6, 2],
            [256, 4, 2],
        ]
        
        input_channel = int(16 * width_mult)
        
        # First convolution layer - adapted for 4 channels (T1, T1ce, T2, FLAIR)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )
        
        # Build RepViT blocks
        self.features = nn.ModuleList()
        for c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(RepViTBlock(input_channel, output_channel, stride))
                input_channel = output_channel
        
        # Cross-modal attention for early fusion
        self.cross_modal_attention = CrossModalAttention(input_channel)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: Input tensor of shape [B, 4, H, W] where 4 channels are T1, T1ce, T2, FLAIR
        """
        x = self.stem(x)
        
        feature_maps = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            # Collect feature maps at different scales for FPN
            if i in [1, 4, 9, 15]:  # Collect at different scales
                feature_maps.append(x)
        
        # Apply cross-modal attention to the final feature map
        x = self.cross_modal_attention(x)
        feature_maps[-1] = x
        
        return feature_maps

class CrossModalAttention(nn.Module):
    """Cross-modal attention for multimodal feature fusion"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        # Generate queries, keys, values
        q = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        k = self.key(x).view(B, -1, W * H)
        v = self.value(x).view(B, -1, W * H)
        
        # Attention
        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        return out

class YOLOv9Neck(nn.Module):
    """YOLOv9 Neck with Feature Pyramid Network"""
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, 1)
            )
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, features):
        # Build FPN
        laterals = [lateral_conv(features[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] += F.interpolate(
                laterals[i + 1], 
                size=laterals[i].shape[-2:], 
                mode='nearest'
            )
        
        # Apply FPN convolutions
        fpn_outs = [fpn_conv(lateral) for lateral, fpn_conv in zip(laterals, self.fpn_convs)]
        
        return fpn_outs

class YOLOv9Head(nn.Module):
    """YOLOv9 Detection Head"""
    def __init__(self, num_classes=1, in_channels=256, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Classification head
        self.cls_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        )
        
        # Regression head (x, y, w, h)
        self.reg_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 4, 1)
        )
        
        # Objectness head
        self.obj_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors, 1)
        )

    def forward(self, x):
        cls_score = self.cls_convs(x)
        bbox_pred = self.reg_convs(x)
        objectness = self.obj_convs(x)
        
        return cls_score, bbox_pred, objectness

class MultimodalPKYOLO(nn.Module):
    """Complete Multimodal PK-YOLO model"""
    def __init__(self, num_classes=1, input_channels=4):
        super().__init__()
        
        # Backbone
        self.backbone = SparKRepViTBackbone(input_channels=input_channels)
        
        # Get backbone output channels (need to calculate based on RepViT structure)
        backbone_channels = [32, 64, 128, 256]  # Adjust based on actual backbone output
        
        # Neck
        self.neck = YOLOv9Neck(backbone_channels)
        
        # Head
        self.head = YOLOv9Head(num_classes=num_classes)
        
        # Anchors for different scales
        self.anchors = torch.tensor([
            [[10, 13], [16, 30], [33, 23]],      # P3/8
            [[30, 61], [62, 45], [59, 119]],     # P4/16
            [[116, 90], [156, 198], [373, 326]]  # P5/32
        ], dtype=torch.float32)

    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        
        # Process through neck
        fpn_features = self.neck(features)
        
        # Generate predictions for each scale
        predictions = []
        for feat in fpn_features:
            cls_score, bbox_pred, objectness = self.head(feat)
            predictions.append((cls_score, bbox_pred, objectness))
        
        return predictions

class BraTSDataset(Dataset):
    """BraTS2020 Dataset for multimodal brain tumor detection"""
    
    def __init__(self, data_dir, split='train', img_size=640, augment=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'
        
        # Get all image files
        self.image_dir = self.data_dir / split / 'images'
        self.label_dir = self.data_dir / split / 'labels'
        
        # Find all unique slice identifiers
        self.slice_ids = self._get_slice_ids()
        
        # Setup augmentations
        self.setup_transforms()
        
        logger.info(f"Loaded {len(self.slice_ids)} samples for {split} split")

    def _get_slice_ids(self):
        """Extract unique slice identifiers from image filenames"""
        slice_ids = set()
        
        for img_file in self.image_dir.glob('*.png'):
            # Extract slice ID from filename (remove modality suffix)
            # Expected format: BraTS20_Training_002_slice_029_t1.png
            filename = img_file.stem
            parts = filename.split('_')
            if len(parts) >= 4:
                slice_id = '_'.join(parts[:-1])  # Remove last part (modality)
                slice_ids.add(slice_id)
        
        return sorted(list(slice_ids))

    def setup_transforms(self):
        """Setup image transformations"""
        if self.augment:
            self.transform = A.Compose([
                A.RandomResizedCrop(self.img_size, self.img_size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomGamma(p=0.3),
                A.GaussNoise(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406, 0.485], 
                           std=[0.229, 0.224, 0.225, 0.229]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406, 0.485], 
                           std=[0.229, 0.224, 0.225, 0.229]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def load_multimodal_image(self, slice_id):
        """Load all 4 modalities for a given slice"""
        modalities = ['t1', 't1ce', 't2', 'flair']
        images = []
        
        for modality in modalities:
            img_path = self.image_dir / f"{slice_id}_{modality}.png"
            if img_path.exists():
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.warning(f"Could not load image: {img_path}")
                    img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                logger.warning(f"Missing modality {modality} for slice {slice_id}")
                img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
            images.append(img)
        
        # Stack to create 4-channel image
        multimodal_img = np.stack(images, axis=-1)  # H, W, 4
        return multimodal_img

    def load_labels(self, slice_id):
        """Load YOLO format labels"""
        label_path = self.label_dir / f"{slice_id}.txt"
        bboxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
        
        return np.array(bboxes, dtype=np.float32), np.array(class_labels, dtype=np.int64)

    def __len__(self):
        return len(self.slice_ids)

    def __getitem__(self, idx):
        slice_id = self.slice_ids[idx]
        
        # Load multimodal image
        image = self.load_multimodal_image(slice_id)
        
        # Load labels
        bboxes, class_labels = self.load_labels(slice_id)
        
        # Apply transformations
        if len(bboxes) > 0:
            transformed = self.transform(
                image=image, 
                bboxes=bboxes.tolist(), 
                class_labels=class_labels.tolist()
            )
            image = transformed['image']
            bboxes = np.array(transformed['bboxes'], dtype=np.float32)
            class_labels = np.array(transformed['class_labels'], dtype=np.int64)
        else:
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed['image']
            bboxes = np.zeros((0, 4), dtype=np.float32)
            class_labels = np.zeros((0,), dtype=np.int64)
        
        return {
            'image': image,
            'bboxes': bboxes,
            'class_labels': class_labels,
            'slice_id': slice_id
        }

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    images = torch.stack([item['image'] for item in batch])
    
    # Handle variable number of bboxes per image
    max_boxes = max(len(item['bboxes']) for item in batch)
    if max_boxes == 0:
        max_boxes = 1
    
    batch_bboxes = torch.zeros(len(batch), max_boxes, 4)
    batch_labels = torch.zeros(len(batch), max_boxes, dtype=torch.long)
    
    for i, item in enumerate(batch):
        if len(item['bboxes']) > 0:
            batch_bboxes[i, :len(item['bboxes'])] = torch.from_numpy(item['bboxes'])
            batch_labels[i, :len(item['class_labels'])] = torch.from_numpy(item['class_labels'])
    
    return {
        'images': images,
        'bboxes': batch_bboxes,
        'labels': batch_labels,
        'slice_ids': [item['slice_id'] for item in batch]
    }

class YOLOLoss(nn.Module):
    """YOLO Loss Function"""
    
    def __init__(self, num_classes=1, ignore_thresh=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets):
        # Implement YOLO loss calculation
        # This is a simplified version - you may want to implement the full YOLOv9 loss
        total_loss = 0
        
        for pred in predictions:
            cls_score, bbox_pred, objectness = pred
            # Calculate loss components
            # obj_loss, cls_loss, bbox_loss = self.calculate_loss(cls_score, bbox_pred, objectness, targets)
            # total_loss += obj_loss + cls_loss + bbox_loss
        
        return total_loss

def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda'):
    """Training loop"""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = YOLOLoss()
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['images'].to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, batch)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                predictions = model(images)
                loss = criterion(predictions, batch)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_multimodal_pk_yolo.pth')
        
        scheduler.step()

def main():
    """Main training function"""
    # Configuration
    config = {
        'data_dir': '/path/to/BraTS2020_TrainingData_ok',
        'batch_size': 8,
        'num_epochs': 100,
        'img_size': 640,
        'num_classes': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create datasets
    train_dataset = BraTSDataset(
        config['data_dir'], 
        split='train', 
        img_size=config['img_size'],
        augment=True
    )
    
    val_dataset = BraTSDataset(
        config['data_dir'], 
        split='test', 
        img_size=config['img_size'],
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Create model
    model = MultimodalPKYOLO(num_classes=config['num_classes'])
    
    # Start training
    train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=config['num_epochs'],
        device=config['device']
    )

if __name__ == "__main__":
    main()