import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from complete_yolo_loss import YOLOLoss, bbox_iou, FocalLoss, smooth_BCE

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
        
        out = out * self.se(out)
        
        if self.identity:
            out += identity
            
        return out

class SparKRepViTBackbone(nn.Module):
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
    """Cross-modal attention for multimodal feature fusion - Memory efficient version"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        # Use smaller spatial dimensions to reduce memory
        if H * W > 64 * 64:  # If spatial size is too large
            # Downsample for attention computation
            x_small = F.adaptive_avg_pool2d(x, (32, 32))  # Fixed small size
            _, _, h_small, w_small = x_small.size()
            
            # Generate queries, keys, values on downsampled feature
            q = self.query(x_small).view(B, -1, w_small * h_small).permute(0, 2, 1)
            k = self.key(x_small).view(B, -1, w_small * h_small)
            v = self.value(x_small).view(B, -1, w_small * h_small)
            
            # Attention
            attention = torch.bmm(q, k)
            attention = F.softmax(attention, dim=-1)
            
            # Apply attention to values
            out_small = torch.bmm(v, attention.permute(0, 2, 1))
            out_small = out_small.view(B, C, h_small, w_small)
            
            # Upsample back to original size
            out = F.interpolate(out_small, size=(H, W), mode='bilinear', align_corners=False)
        else:
            # Original attention for small feature maps
            q = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
            k = self.key(x).view(B, -1, W * H)
            v = self.value(x).view(B, -1, W * H)
            
            attention = torch.bmm(q, k)
            attention = F.softmax(attention, dim=-1)
            
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
    """Complete Multimodal PK-YOLO model with proper loss integration"""
    def __init__(self, num_classes=1, input_channels=4):
        super().__init__()
        
        # Model configuration
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Backbone
        self.backbone = SparKRepViTBackbone(input_channels=input_channels)
        
        # Get backbone output channels (need to calculate based on RepViT structure)
        backbone_channels = [32, 64, 128, 256]  # Adjust based on actual backbone output
        
        # Neck
        self.neck = YOLOv9Neck(backbone_channels)
        
        # Head
        self.head = YOLOv9Head(num_classes=num_classes)
        
        # Anchors for different scales (optimized for brain tumor sizes)
        self.anchors = torch.tensor([
            [[10, 13], [16, 30], [33, 23]],      # P3/8 - small tumors
            [[30, 61], [62, 45], [59, 119]],     # P4/16 - medium tumors  
            [[116, 90], [156, 198], [373, 326]]  # P5/32 - large tumors
        ], dtype=torch.float32)
        
        # Hyperparameters for training (used by loss function)
        self.hyp = {
            'box': 0.05,           # Box loss weight
            'cls': 0.5,            # Class loss weight  
            'obj': 1.0,            # Object loss weight
            'anchor_t': 4.0,       # Anchor matching threshold
            'fl_gamma': 0.0,       # Focal loss gamma (0 = no focal loss)
            'cls_pw': 1.0,         # Class positive weight
            'obj_pw': 1.0,         # Object positive weight
            'label_smoothing': 0.0, # Label smoothing epsilon
        }

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
        
        # Look for all PNG files in the images directory
        image_files = list(self.image_dir.glob('*.png'))
        
        for img_file in image_files:
            filename = img_file.stem
            
            # Check if this is a modality-specific file (ends with _t1, _t1ce, _t2, _flair)
            if filename.endswith(('_t1', '_t1ce', '_t2', '_flair')):
                # Remove the modality suffix to get the slice ID
                if filename.endswith('_t1ce'):
                    slice_id = filename[:-5]  # Remove '_t1ce'
                elif filename.endswith('_flair'):
                    slice_id = filename[:-6]  # Remove '_flair'
                else:
                    slice_id = filename[:-3]  # Remove '_t1' or '_t2'
                
                slice_ids.add(slice_id)
            else:
                # If no modality suffix found, use the filename as is
                slice_ids.add(filename)
        
        if not slice_ids:
            logger.warning(f"No slice IDs found in {self.image_dir}")
            logger.info(f"Available files: {[f.name for f in image_files[:10]]}")  # Show first 10 files
        
        return sorted(list(slice_ids))

    def setup_transforms(self):
        """Setup image transformations"""
        if self.augment:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
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
    images = torch.stack([item['image'] for item in batch]).float()  # Ensure float32
    
    # Handle variable number of bboxes per image
    max_boxes = max(len(item['bboxes']) for item in batch)
    if max_boxes == 0:
        max_boxes = 1
    
    batch_bboxes = torch.zeros(len(batch), max_boxes, 4, dtype=torch.float32)
    batch_labels = torch.zeros(len(batch), max_boxes, dtype=torch.long)
    
    for i, item in enumerate(batch):
        if len(item['bboxes']) > 0:
            batch_bboxes[i, :len(item['bboxes'])] = torch.from_numpy(item['bboxes']).float()
            batch_labels[i, :len(item['class_labels'])] = torch.from_numpy(item['class_labels']).long()
    
    return {
        'images': images,
        'bboxes': batch_bboxes,
        'labels': batch_labels,
        'slice_ids': [item['slice_id'] for item in batch]
    }

def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda'):
    """Enhanced training loop with complete YOLO loss function"""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Initialize the complete YOLO loss
    criterion = YOLOLoss(model, num_classes=1, autobalance=True)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_box_loss = 0.0
        train_obj_loss = 0.0
        train_cls_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['images'].to(device)
            
            # Move target data to device
            targets = {
                'bboxes': batch['bboxes'].to(device),
                'labels': batch['labels'].to(device),
                'images': images
            }
            
            optimizer.zero_grad()
            predictions = model(images)
            
            # Calculate loss using the complete YOLO loss function
            total_loss, loss_components = criterion(predictions, targets)
            
            total_loss.backward()
            optimizer.step()
            
            # Track losses
            train_loss += total_loss.item()
            train_box_loss += loss_components[0].item()
            train_obj_loss += loss_components[1].item()
            train_cls_loss += loss_components[2].item()
            
            if batch_idx % 50 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, '
                          f'Total Loss: {total_loss.item():.4f}, '
                          f'Box: {loss_components[0].item():.4f}, '
                          f'Obj: {loss_components[1].item():.4f}, '
                          f'Cls: {loss_components[2].item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_box_loss = 0.0
        val_obj_loss = 0.0
        val_cls_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                targets = {
                    'bboxes': batch['bboxes'].to(device),
                    'labels': batch['labels'].to(device),
                    'images': images
                }
                
                predictions = model(images)
                total_loss, loss_components = criterion(predictions, targets)
                
                val_loss += total_loss.item()
                val_box_loss += loss_components[0].item()
                val_obj_loss += loss_components[1].item()
                val_cls_loss += loss_components[2].item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f'Epoch {epoch}: '
                   f'Train Loss: {avg_train_loss:.4f} '
                   f'(Box: {train_box_loss/len(train_loader):.4f}, '
                   f'Obj: {train_obj_loss/len(train_loader):.4f}, '
                   f'Cls: {train_cls_loss/len(train_loader):.4f}), '
                   f'Val Loss: {avg_val_loss:.4f} '
                   f'(Box: {val_box_loss/len(val_loader):.4f}, '
                   f'Obj: {val_obj_loss/len(val_loader):.4f}, '
                   f'Cls: {val_cls_loss/len(val_loader):.4f})')
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'anchors': model.anchors,
                'hyp': model.hyp
            }, 'best_multimodal_pk_yolo.pth')
            logger.info(f'New best model saved with validation loss: {avg_val_loss:.4f}')
        
        scheduler.step()


def main():
    """Main training function with complete loss integration"""
    # Configuration
    config = {
        'data_dir': '/path/to/BraTS2020_TrainingData_ok',
        'batch_size': 8,
        'num_epochs': 100,
        'img_size': 640,
        'num_classes': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info(f"Using device: {config['device']}")
    logger.info(f"Training for {config['num_epochs']} epochs with batch size {config['batch_size']}")
    
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
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = MultimodalPKYOLO(
        num_classes=config['num_classes'], 
        input_channels=4  # T1, T1ce, T2, FLAIR
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Start training with complete loss function
    train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=config['num_epochs'],
        device=config['device']
    )


# def load_pretrained_model(checkpoint_path, device='cuda'):
#     """Load a pretrained multimodal PK-YOLO model"""
#     checkpoint = torch.load(checkpoint_path, map_location=device)
    
#     model = MultimodalPKYOLO(num_classes=1, input_channels=4)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model = model.to(device)
#     model.eval()
    
#     logger.info(f"Loaded pretrained model from {checkpoint_path}")
#     logger.info(f"Best validation loss: {checkpoint['loss']:.4f}")
    
#     return model


# def predict_brain_tumors(model, image_tensor, device='cuda', conf_threshold=0.5):
#     """
#     Predict brain tumors from a 4-channel MRI image
#     Args:
#         model: Trained MultimodalPKYOLO model
#         image_tensor: Input tensor of shape [1, 4, H, W] (T1, T1ce, T2, FLAIR)
#         device: Device to run inference on
#         conf_threshold: Confidence threshold for detections
#     Returns:
#         List of detections [x1, y1, x2, y2, confidence, class]
#     """
#     model.eval()
#     image_tensor = image_tensor.to(device)
    
#     with torch.no_grad():
#         predictions = model(image_tensor)
        
#         # Post-process predictions to get final detections
#         detections = post_process_predictions(
#             predictions, 
#             conf_threshold=conf_threshold,
#             image_size=image_tensor.shape[-2:]
#         )
    
#     return detections


# def post_process_predictions(predictions, conf_threshold=0.5, nms_threshold=0.4, image_size=(640, 640)):
#     """
#     Post-process YOLO predictions to get final detections
#     Args:
#         predictions: Raw model predictions
#         conf_threshold: Confidence threshold
#         nms_threshold: NMS threshold
#         image_size: Original image size
#     Returns:
#         Final detections after NMS
#     """
#     detections = []
    
#     for i, (cls_score, bbox_pred, objectness) in enumerate(predictions):
#         # Get grid size
#         grid_h, grid_w = cls_score.shape[-2:]
        
#         # Create grid coordinates
#         grid_y, grid_x = torch.meshgrid(
#             torch.arange(grid_h), 
#             torch.arange(grid_w), 
#             indexing='ij'
#         )
#         grid_y = grid_y.to(cls_score.device).float()
#         grid_x = grid_x.to(cls_score.device).float()
        
#         # Reshape predictions
#         batch_size = cls_score.shape[0]
#         num_anchors = 3
        
#         cls_score = cls_score.view(batch_size, num_anchors, -1, grid_h, grid_w).permute(0, 1, 3, 4, 2)
#         bbox_pred = bbox_pred.view(batch_size, num_anchors, 4, grid_h, grid_w).permute(0, 1, 3, 4, 2)
#         objectness = objectness.view(batch_size, num_anchors, 1, grid_h, grid_w).permute(0, 1, 3, 4, 2)
        
#         # Apply sigmoid and process coordinates
#         obj_conf = torch.sigmoid(objectness)
#         cls_conf = torch.sigmoid(cls_score)
        
#         # Convert bbox predictions to actual coordinates
#         stride = image_size[0] // grid_h  # Assuming square images
        
#         # Box center coordinates
#         box_xy = (torch.sigmoid(bbox_pred[..., :2]) * 1.6 - 0.3 + torch.stack([grid_x, grid_y], dim=-1)) * stride
        
#         # Box width and height  
#         anchors = torch.tensor([
#             [10, 13], [16, 30], [33, 23]  # Adjust based on scale
#         ], device=cls_score.device).float()
        
#         if i == 1:  # P4
#             anchors = torch.tensor([[30, 61], [62, 45], [59, 119]], device=cls_score.device).float()
#         elif i == 2:  # P5
#             anchors = torch.tensor([[116, 90], [156, 198], [373, 326]], device=cls_score.device).float()
        
#         box_wh = (0.2 + torch.sigmoid(bbox_pred[..., 2:4]) * 4.8) * anchors.view(1, -1, 1, 1, 2)
        
#         # Combine confidence scores
#         conf = obj_conf * cls_conf
        
#         # Filter by confidence threshold
#         conf_mask = conf.squeeze(-1) > conf_threshold
        
#         if conf_mask.any():
#             # Convert to [x1, y1, x2, y2] format
#             box_x1y1 = box_xy - box_wh / 2
#             box_x2y2 = box_xy + box_wh / 2
#             boxes = torch.cat([box_x1y1, box_x2y2], dim=-1)
            
#             # Get valid detections
#             valid_boxes = boxes[conf_mask]
#             valid_conf = conf[conf_mask]
            
#             detections.append({
#                 'boxes': valid_boxes,
#                 'scores': valid_conf,
#                 'scale_idx': i
#             })
    
#     return detections


# def evaluate_model(model, val_loader, device='cuda'):
#     """
#     Evaluate the model on validation set
#     Args:
#         model: Trained model
#         val_loader: Validation data loader
#         device: Device to run evaluation on
#     Returns:
#         Evaluation metrics
#     """
#     model.eval()
#     criterion = YOLOLoss(model, num_classes=1, autobalance=False)
    
#     total_loss = 0.0
#     total_box_loss = 0.0
#     total_obj_loss = 0.0
#     total_cls_loss = 0.0
#     num_batches = 0
    
#     with torch.no_grad():
#         for batch in val_loader:
#             images = batch['images'].to(device)
#             targets = {
#                 'bboxes': batch['bboxes'].to(device),
#                 'labels': batch['labels'].to(device),
#                 'images': images
#             }
            
#             predictions = model(images)
#             loss, loss_components = criterion(predictions, targets)
            
#             total_loss += loss.item()
#             total_box_loss += loss_components[0].item()
#             total_obj_loss += loss_components[1].item()
#             total_cls_loss += loss_components[2].item()
#             num_batches += 1
    
#     avg_loss = total_loss / num_batches
#     avg_box_loss = total_box_loss / num_batches
#     avg_obj_loss = total_obj_loss / num_batches
#     avg_cls_loss = total_cls_loss / num_batches
    
#     logger.info(f"Validation Results:")
#     logger.info(f"Average Total Loss: {avg_loss:.4f}")
#     logger.info(f"Average Box Loss: {avg_box_loss:.4f}")
#     logger.info(f"Average Objectness Loss: {avg_obj_loss:.4f}")
#     logger.info(f"Average Classification Loss: {avg_cls_loss:.4f}")
    
#     return {
#         'total_loss': avg_loss,
#         'box_loss': avg_box_loss,
#         'obj_loss': avg_obj_loss,
#         'cls_loss': avg_cls_loss
#     }

# def apply_multimodal_preprocessing(t1_path, t1ce_path, t2_path, flair_path, target_size=(640, 640)):
#     """
#     Preprocess 4-channel MRI data for the model
#     Args:
#         t1_path, t1ce_path, t2_path, flair_path: Paths to individual modality images
#         target_size: Target image size
#     Returns:
#         Preprocessed 4-channel tensor
#     """
#     modalities = []
    
#     for path in [t1_path, t1ce_path, t2_path, flair_path]:
#         img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             raise ValueError(f"Could not load image: {path}")
        
#         # Resize to target size
#         img = cv2.resize(img, target_size)
        
#         # Normalize to [0, 1]
#         img = img.astype(np.float32) / 255.0
        
#         modalities.append(img)
    
#     # Stack modalities and create tensor
#     multimodal_img = np.stack(modalities, axis=0)  # [4, H, W]
    
#     # Apply normalization (same as training)
#     mean = np.array([0.485, 0.456, 0.406, 0.485])
#     std = np.array([0.229, 0.224, 0.225, 0.229])
    
#     for i in range(4):
#         multimodal_img[i] = (multimodal_img[i] - mean[i]) / std[i]
    
#     return torch.tensor(multimodal_img).unsqueeze(0)  # Add batch dimension


# def visualize_detections(image_tensor, detections, save_path=None):
#     """
#     Visualize detections on the input image
#     Args:
#         image_tensor: Input 4-channel tensor [1, 4, H, W]
#         detections: List of detections
#         save_path: Path to save visualization
#     """
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as patches
    
#     # Use T1ce modality for visualization (channel 1)
#     img = image_tensor[0, 1].cpu().numpy()
    
#     # Denormalize for visualization
#     img = img * 0.224 + 0.456  # Reverse normalization
#     img = np.clip(img, 0, 1)
    
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     ax.imshow(img, cmap='gray')
    
#     # Draw detections
#     for det_group in detections:
#         if 'boxes' in det_group:
#             boxes = det_group['boxes'].cpu().numpy()
#             scores = det_group['scores'].cpu().numpy()
            
#             for box, score in zip(boxes, scores):
#                 x1, y1, x2, y2 = box
#                 width = x2 - x1
#                 height = y2 - y1
                
#                 # Draw bounding box
#                 rect = patches.Rectangle(
#                     (x1, y1), width, height, 
#                     linewidth=2, edgecolor='red', facecolor='none'
#                 )
#                 ax.add_patch(rect)
                
#                 # Add confidence score
#                 ax.text(x1, y1-5, f'{score:.2f}', 
#                        color='red', fontsize=12, fontweight='bold')
    
#     ax.set_title('Brain Tumor Detection Results')
#     ax.axis('off')
    
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=300)
#         logger.info(f"Visualization saved to {save_path}")
    
#     plt.show()


# if __name__ == "__main__":
#     main()

"""
Multimodal PK-YOLO Implementation for Brain Tumor Detection
4-Channel Backbone Setup with RepViT + SparK + YOLOv9
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from pathlib import Path

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
        
        out = out * self.se(out)
        
        if self.identity:
            out += identity
            
        return out

class SparKRepViTBackbone(nn.Module):
    """SparK-pretrained RepViT backbone adapted for 4-channel multimodal input"""
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
    """Cross-modal attention for multimodal feature fusion - Memory efficient version"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        # Use smaller spatial dimensions to reduce memory
        if H * W > 64 * 64:  # If spatial size is too large
            # Downsample for attention computation
            x_small = F.adaptive_avg_pool2d(x, (32, 32))  # Fixed small size
            _, _, h_small, w_small = x_small.size()
            
            # Generate queries, keys, values on downsampled feature
            q = self.query(x_small).view(B, -1, w_small * h_small).permute(0, 2, 1)
            k = self.key(x_small).view(B, -1, w_small * h_small)
            v = self.value(x_small).view(B, -1, w_small * h_small)
            
            # Attention
            attention = torch.bmm(q, k)
            attention = F.softmax(attention, dim=-1)
            
            # Apply attention to values
            out_small = torch.bmm(v, attention.permute(0, 2, 1))
            out_small = out_small.view(B, C, h_small, w_small)
            
            # Upsample back to original size
            out = F.interpolate(out_small, size=(H, W), mode='bilinear', align_corners=False)
        else:
            # Original attention for small feature maps
            q = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
            k = self.key(x).view(B, -1, W * H)
            v = self.value(x).view(B, -1, W * H)
            
            attention = torch.bmm(q, k)
            attention = F.softmax(attention, dim=-1)
            
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
        
        # Model configuration
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Backbone
        self.backbone = SparKRepViTBackbone(input_channels=input_channels)
        
        # Get backbone output channels (calculated based on RepViT structure)
        backbone_channels = [32, 64, 128, 256]  # Adjust based on actual backbone output
        
        # Neck
        self.neck = YOLOv9Neck(backbone_channels)
        
        # Head
        self.head = YOLOv9Head(num_classes=num_classes)
        
        # Anchors for different scales (optimized for brain tumor sizes)
        self.anchors = torch.tensor([
            [[10, 13], [16, 30], [33, 23]],      # P3/8 - small tumors
            [[30, 61], [62, 45], [59, 119]],     # P4/16 - medium tumors  
            [[116, 90], [156, 198], [373, 326]]  # P5/32 - large tumors
        ], dtype=torch.float32)
        
        # Hyperparameters for training (used by loss function)
        self.hyp = {
            'box': 0.05,           # Box loss weight
            'cls': 0.5,            # Class loss weight  
            'obj': 1.0,            # Object loss weight
            'anchor_t': 4.0,       # Anchor matching threshold
            'fl_gamma': 0.0,       # Focal loss gamma (0 = no focal loss)
            'cls_pw': 1.0,         # Class positive weight
            'obj_pw': 1.0,         # Object positive weight
            'label_smoothing': 0.0, # Label smoothing epsilon
        }

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
        
        # Look for all PNG files in the images directory
        image_files = list(self.image_dir.glob('*.png'))
        
        for img_file in image_files:
            filename = img_file.stem
            
            # Check if this is a modality-specific file (ends with _t1, _t1ce, _t2, _flair)
            if filename.endswith(('_t1', '_t1ce', '_t2', '_flair')):
                # Remove the modality suffix to get the slice ID
                if filename.endswith('_t1ce'):
                    slice_id = filename[:-5]  # Remove '_t1ce'
                elif filename.endswith('_flair'):
                    slice_id = filename[:-6]  # Remove '_flair'
                else:
                    slice_id = filename[:-3]  # Remove '_t1' or '_t2'
                
                slice_ids.add(slice_id)
            else:
                # If no modality suffix found, use the filename as is
                slice_ids.add(filename)
        
        if not slice_ids:
            logger.warning(f"No slice IDs found in {self.image_dir}")
            logger.info(f"Available files: {[f.name for f in image_files[:10]]}")  # Show first 10 files
        
        return sorted(list(slice_ids))

    def setup_transforms(self):
        """Setup image transformations"""
        if self.augment:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
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
    images = torch.stack([item['image'] for item in batch]).float()  # Ensure float32
    
    # Handle variable number of bboxes per image
    max_boxes = max(len(item['bboxes']) for item in batch)
    if max_boxes == 0:
        max_boxes = 1
    
    batch_bboxes = torch.zeros(len(batch), max_boxes, 4, dtype=torch.float32)
    batch_labels = torch.zeros(len(batch), max_boxes, dtype=torch.long)
    
    for i, item in enumerate(batch):
        if len(item['bboxes']) > 0:
            batch_bboxes[i, :len(item['bboxes'])] = torch.from_numpy(item['bboxes']).float()
            batch_labels[i, :len(item['class_labels'])] = torch.from_numpy(item['class_labels']).long()
    
    return {
        'images': images,
        'bboxes': batch_bboxes,
        'labels': batch_labels,
        'slice_ids': [item['slice_id'] for item in batch]
    }

# Utility functions for model loading and inference
def load_pretrained_model(checkpoint_path, device='cuda'):
    """Load a pretrained multimodal PK-YOLO model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = MultimodalPKYOLO(num_classes=1, input_channels=4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded pretrained model from {checkpoint_path}")
    if 'loss' in checkpoint:
        logger.info(f"Best validation loss: {checkpoint['loss']:.4f}")
    
    return model

def create_model(num_classes=1, input_channels=4, pretrained_path=None, device='cuda'):
    """Create a MultimodalPKYOLO model with optional pretrained weights"""
    model = MultimodalPKYOLO(num_classes=num_classes, input_channels=input_channels)
    
    if pretrained_path and Path(pretrained_path).exists():
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded pretrained weights from {pretrained_path}")
    
    return model.to(device)

def preprocess_multimodal_image(t1_path, t1ce_path, t2_path, flair_path, target_size=(640, 640)):
    """
    Preprocess 4-channel MRI data for the model
    Args:
        t1_path, t1ce_path, t2_path, flair_path: Paths to individual modality images
        target_size: Target image size
    Returns:
        Preprocessed 4-channel tensor
    """
    modalities = []
    
    for path in [t1_path, t1ce_path, t2_path, flair_path]:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        
        # Resize to target size
        img = cv2.resize(img, target_size)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        modalities.append(img)
    
    # Stack modalities and create tensor
    multimodal_img = np.stack(modalities, axis=0)  # [4, H, W]
    
    # Apply normalization (same as training)
    mean = np.array([0.485, 0.456, 0.406, 0.485])
    std = np.array([0.229, 0.224, 0.225, 0.229])
    
    for i in range(4):
        multimodal_img[i] = (multimodal_img[i] - mean[i]) / std[i]
    
    return torch.tensor(multimodal_img).unsqueeze(0)  # Add batch dimension

def get_model_info(model):
    """Get model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'input_channels': model.input_channels,
        'num_classes': model.num_classes
    }
    
    return info

if __name__ == "__main__":
    # Example usage
    print("Multimodal PK-YOLO model ready!")
    print("To train the model, use: python training_script.py --data_dir /path/to/data")
    print("To use inference utilities, use: python config_and_inference.py")
    
    # Create a sample model and show info
    model = MultimodalPKYOLO(num_classes=1, input_channels=4)
    info = get_model_info(model)
    
    print(f"\nModel Information:")
    print(f"  Total parameters: {info['total_parameters']:,}")
    print(f"  Trainable parameters: {info['trainable_parameters']:,}")
    print(f"  Model size: {info['model_size_mb']:.2f} MB")
    print(f"  Input channels: {info['input_channels']}")
    print(f"  Number of classes: {info['num_classes']}")