"""
Fixed Multimodal PK-YOLO Model
Properly structured for 4-channel brain tumor detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

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
        
        # Squeeze-and-Excitation attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(oup, max(1, oup // 16), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, oup // 16), oup, 1),
            nn.Sigmoid()
        )
        
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        if self.identity:
            identity = x
            
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.dwconv(out)))
        out = self.bn3(self.conv2(out))
        
        # Apply SE attention
        out = out * self.se(out)
        
        if self.identity:
            out += identity
            
        return out

class CrossModalAttention(nn.Module):
    """Memory efficient cross-modal attention"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        # For large spatial sizes, use downsampling for efficiency
        if H * W > 64 * 64:
            # Downsample for attention computation
            x_small = F.adaptive_avg_pool2d(x, (32, 32))
            _, _, h_small, w_small = x_small.size()
            
            q = self.query(x_small).view(B, -1, w_small * h_small).permute(0, 2, 1)
            k = self.key(x_small).view(B, -1, w_small * h_small)
            v = self.value(x_small).view(B, -1, w_small * h_small)
            
            attention = torch.bmm(q, k)
            attention = F.softmax(attention, dim=-1)
            
            out_small = torch.bmm(v, attention.permute(0, 2, 1))
            out_small = out_small.view(B, C, h_small, w_small)
            
            # Upsample back to original size
            out = F.interpolate(out_small, size=(H, W), mode='bilinear', align_corners=False)
        else:
            # Direct attention for smaller feature maps
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

class SparKRepViTBackbone(nn.Module):
    """RepViT backbone adapted for 4-channel multimodal input"""
    def __init__(self, input_channels=4, width_mult=1.0):
        super().__init__()
        
        # Layer configurations: [channels, num_blocks, stride]
        self.cfgs = [
            [32, 1, 1],   # Stage 1
            [64, 2, 2],   # Stage 2
            [128, 4, 2],  # Stage 3
            [256, 6, 2],  # Stage 4
            [512, 4, 2],  # Stage 5
        ]
        
        input_channel = int(32 * width_mult)
        
        # Stem layer - adapted for 4 channels (T1, T1ce, T2, FLAIR)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )
        
        # Build RepViT blocks
        self.stages = nn.ModuleList()
        stage_idx = 0
        
        for c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            stage_blocks = nn.ModuleList()
            
            for i in range(n):
                stride = s if i == 0 else 1
                stage_blocks.append(RepViTBlock(input_channel, output_channel, stride))
                input_channel = output_channel
            
            self.stages.append(stage_blocks)
            stage_idx += 1
        
        # Cross-modal attention for the final stage
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
        """Forward pass through backbone"""
        x = self.stem(x)
        
        feature_maps = []
        
        # Pass through each stage
        for stage_idx, stage_blocks in enumerate(self.stages):
            for block in stage_blocks:
                x = block(x)
            
            # Collect feature maps from stages 2, 3, 4, 5 for FPN
            if stage_idx >= 1:  # Skip stage 0 (too early)
                feature_maps.append(x)
        
        # Apply cross-modal attention to final feature map
        if len(feature_maps) > 0:
            feature_maps[-1] = self.cross_modal_attention(feature_maps[-1])
        
        return feature_maps

class YOLOv9Neck(nn.Module):
    """Feature Pyramid Network for YOLO"""
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            # Lateral connections
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, 1)
            )
            # FPN output convolutions
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, features):
        # Build laterals
        laterals = [lateral_conv(features[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample higher resolution feature
            upsampled = F.interpolate(
                laterals[i + 1], 
                size=laterals[i].shape[-2:], 
                mode='nearest'
            )
            laterals[i] += upsampled
        
        # Apply FPN convolutions
        fpn_outs = [fpn_conv(lateral) for lateral, fpn_conv in zip(laterals, self.fpn_convs)]
        
        return fpn_outs

class YOLOv9Head(nn.Module):
    """YOLO detection head"""
    def __init__(self, num_classes=1, in_channels=256, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Shared convolution layers
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        self.cls_convs = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        
        # Regression head (x, y, w, h)
        self.reg_convs = nn.Conv2d(in_channels, num_anchors * 4, 1)
        
        # Objectness head
        self.obj_convs = nn.Conv2d(in_channels, num_anchors, 1)
        
        # Initialize weights
        self._initialize_head()

    def _initialize_head(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize objectness bias for better convergence
        nn.init.constant_(self.obj_convs.bias, -4.6)  # Prior probability of 0.01

    def forward(self, x):
        # Shared feature extraction
        x = self.shared_conv(x)
        
        # Generate predictions
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
        
        # Calculate backbone output channels based on the RepViT structure
        backbone_channels = [64, 128, 256, 512]  # Channels from stages 2, 3, 4, 5
        
        # Neck (Feature Pyramid Network)
        self.neck = YOLOv9Neck(backbone_channels, out_channels=256)
        
        # Detection head
        self.head = YOLOv9Head(num_classes=num_classes, in_channels=256)
        
        # Anchors for different scales (optimized for brain tumor detection)
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
        """Forward pass through the model"""
        # Extract features using backbone
        features = self.backbone(x)
        
        # Process through neck (FPN)
        fpn_features = self.neck(features)
        
        # Generate predictions for each scale
        predictions = []
        for feat in fpn_features:
            cls_score, bbox_pred, objectness = self.head(feat)
            predictions.append((cls_score, bbox_pred, objectness))
        
        return predictions

def get_model_info(model):
    """Get detailed model information"""
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

def create_model(num_classes=1, input_channels=4, pretrained_path=None, device='cuda'):
    """Create a MultimodalPKYOLO model with optional pretrained weights"""
    model = MultimodalPKYOLO(num_classes=num_classes, input_channels=input_channels)
    
    if pretrained_path and Path(pretrained_path).exists():
        checkpoint = torch.load(pretrained_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded pretrained weights from {pretrained_path}")
    
    return model.to(device)

def test_model(input_channels=4, num_classes=1, img_size=640, batch_size=2):
    """Test the model with dummy input"""
    logger.info("Testing model architecture...")
    
    model = MultimodalPKYOLO(num_classes=num_classes, input_channels=input_channels)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, input_channels, img_size, img_size)
    
    try:
        with torch.no_grad():
            predictions = model(dummy_input)
        
        logger.info(f"Model test successful!")
        logger.info(f"Input shape: {dummy_input.shape}")
        logger.info(f"Number of prediction scales: {len(predictions)}")
        
        for i, (cls, bbox, obj) in enumerate(predictions):
            logger.info(f"Scale {i}: cls={cls.shape}, bbox={bbox.shape}, obj={obj.shape}")
        
        # Model info
        info = get_model_info(model)
        logger.info(f"Total parameters: {info['total_parameters']:,}")
        logger.info(f"Model size: {info['model_size_mb']:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test the model
    success = test_model()
    if success:
        print("✅ Model architecture test passed!")
    else:
        print("❌ Model architecture test failed!")