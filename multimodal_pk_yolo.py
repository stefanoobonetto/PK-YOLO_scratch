import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DWConv(nn.Module):
    """Depthwise convolution."""
    def __init__(self, dim, kernel_size=3, stride=1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, 
                                kernel_size//2, groups=dim, bias=False)

    def forward(self, x):
        return self.dwconv(x)


class RepViTBlock(nn.Module):
    """RepViT block with SE attention."""
    def __init__(self, inp, oup, stride=1, expand_ratio=4):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res = stride == 1 and inp == oup
        
        self.conv = nn.Sequential(
            # Expand
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Depthwise
            DWConv(hidden_dim, stride=stride),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # SE
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, hidden_dim, 1),
            nn.Sigmoid(),
            # Project
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        )
        
        # Simplified SE application
        self.se_idx = 6  # Index where SE starts

    def forward(self, x):
        identity = x if self.use_res else None
        
        # Split at SE
        for i, layer in enumerate(self.conv[:self.se_idx]):
            x = layer(x)
        
        # SE attention
        se = x
        for layer in self.conv[self.se_idx:self.se_idx+4]:
            se = layer(se)
        x = x * se
        
        # Final projection
        x = self.conv[-2:](x)
        
        if identity is not None:
            x = x + identity
        return x


class Backbone(nn.Module):
    """Optimized RepViT backbone for small object detection."""
    
    def __init__(self, in_channels=4, width_mult=1.0, spark_path=None):
        super().__init__()
        
        # Adjusted for better small object detection
        # Reduced downsampling in early stages
        self.cfgs = [
            # c, n, s (channels, num_blocks, stride)
            [32, 2, 1],   # Keep higher resolution early
            [64, 3, 2],   # P2 feature
            [128, 3, 2],  # P3 feature  
            [256, 4, 2],  # P4 feature
            [512, 2, 2],  # P5 feature
        ]
        
        # Initial stem with less aggressive downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),  # No stride
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 32, 3, 2, 1, bias=False),  # Single downsample
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # Build stages
        in_c = 32
        self.stages = nn.ModuleList()
        
        for cfg in self.cfgs:
            out_c, n, s = cfg
            out_c = int(out_c * width_mult)
            blocks = []
            
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(RepViTBlock(in_c, out_c, stride))
                in_c = out_c
            
            self.stages.append(nn.Sequential(*blocks))
        
        # Cross-modal fusion
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, in_c // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c // 8, in_c, 1),
            nn.Sigmoid()
        )
        
        if spark_path and Path(spark_path).exists():
            self._load_spark_weights(spark_path)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _load_spark_weights(self, path):
        """Load SparK pretrained weights."""
        try:
            ckpt = torch.load(path, map_location='cpu')
            state = ckpt.get('backbone_state_dict', ckpt.get('backbone', ckpt))
            
            # Load matching weights only
            current = self.state_dict()
            matched = {k: v for k, v in state.items() 
                      if k in current and current[k].shape == v.shape}
            
            current.update(matched)
            self.load_state_dict(current)
            logger.info(f"Loaded {len(matched)}/{len(state)} SparK weights")
        except Exception as e:
            logger.warning(f"Failed loading SparK weights: {e}")

    def forward(self, x):
        x = self.stem(x)
        
        # Collect multi-scale features (P2, P3, P4, P5)
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i >= 1:  # Skip first stage, collect P2-P5
                if i == len(self.stages) - 1:
                    # Apply channel attention to final feature
                    att = self.channel_attention(x)
                    x = x * att
                features.append(x)
        
        return features


class FPN(nn.Module):
    """Enhanced FPN for small object detection."""
    
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        
        # Lateral connections
        self.laterals = nn.ModuleList([
            nn.Conv2d(in_c, out_channels, 1) for in_c in in_channels_list
        ])
        
        # Output convolutions with deformable-like attention
        self.outputs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                # Extra refinement for small objects
                nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in in_channels_list
        ])
        
        # Bottom-up path augmentation for small objects
        self.bu_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, 2, 1)
            for _ in range(len(in_channels_list) - 1)
        ])

    def forward(self, features):
        # Top-down path
        laterals = [lat(f) for lat, f in zip(self.laterals, features)]
        
        # Build top-down
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], 
                mode='bilinear', align_corners=False
            )
        
        # Refine outputs
        outputs = [out(lat) for out, lat in zip(self.outputs, laterals)]
        
        # Bottom-up path augmentation (PAN-like)
        for i in range(len(outputs) - 1):
            outputs[i + 1] = outputs[i + 1] + self.bu_convs[i](outputs[i])
        
        return outputs


class YOLOHead(nn.Module):
    """YOLO head optimized for small tumors."""
    
    def __init__(self, in_channels=256, num_anchors=3):
        super().__init__()
        mid_channels = max(128, in_channels // 2)
        
        self.detect = nn.Sequential(
            # Shared convolutions with more capacity
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Add one more layer for better small object features
            nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Separate heads
        self.box_head = nn.Conv2d(in_channels, num_anchors * 4, 1)
        self.obj_head = nn.Conv2d(in_channels, num_anchors, 1)
        
        # Initialize
        nn.init.normal_(self.box_head.weight, 0, 0.01)
        nn.init.normal_(self.obj_head.weight, 0, 0.01)
        nn.init.constant_(self.obj_head.bias, -4.0)  # Lower initial objectness

    def forward(self, x):
        x = self.detect(x)
        return self.box_head(x), self.obj_head(x)


class MultimodalPKYOLO(nn.Module):
    """Optimized PK-YOLO for small brain tumor detection."""
    
    def __init__(self, input_channels=4, num_classes=1, 
                 use_spark_pretrained=False, spark_pretrained_path=None):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_spark_pretrained = use_spark_pretrained
        
        # Backbone
        self.backbone = Backbone(
            in_channels=input_channels,
            spark_path=spark_pretrained_path if use_spark_pretrained else None
        )
        
        # Get channel sizes from backbone output
        # P2=64, P3=128, P4=256, P5=512
        backbone_channels = [64, 128, 256, 512]
        
        # FPN neck
        self.neck = FPN(backbone_channels, out_channels=256)
        
        # Detection heads
        self.heads = nn.ModuleList([
            YOLOHead(256, num_anchors=3) for _ in range(4)  # P2-P5
        ])
        
        # OPTIMIZED ANCHORS FOR SMALL TUMORS
        # Calculated from your dataset statistics
        self.register_buffer('anchors', torch.tensor([
            # P2 (8x downsample) - tiny tumors
            [[4, 4], [8, 7], [13, 11]],
            # P3 (16x downsample) - small tumors  
            [[19, 17], [28, 24], [41, 35]],
            # P4 (32x downsample) - medium tumors
            [[59, 52], [87, 74], [122, 98]],
            # P5 (64x downsample) - large tumors
            [[165, 141], [234, 187], [340, 280]]
        ], dtype=torch.float32))

    def forward(self, x):
        # Multi-scale features
        features = self.backbone(x)
        
        # FPN refinement
        fpn_outs = self.neck(features)
        
        # Detection at each scale
        predictions = []
        for fpn_feat, head in zip(fpn_outs, self.heads):
            bbox_pred, obj_pred = head(fpn_feat)
            predictions.append((bbox_pred, obj_pred))
        
        return predictions


def create_model(num_classes=1, input_channels=4, pretrained_path=None, 
                device='cuda', use_spark_pretrained=False, 
                spark_pretrained_path=None):
    """Create optimized model."""
    
    model = MultimodalPKYOLO(
        input_channels=input_channels,
        num_classes=num_classes,
        use_spark_pretrained=use_spark_pretrained,
        spark_pretrained_path=spark_pretrained_path
    )
    
    if pretrained_path and Path(pretrained_path).exists():
        ckpt = torch.load(pretrained_path, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            logger.info(f"Loaded model from epoch {ckpt.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(ckpt)
    
    return model.to(device)