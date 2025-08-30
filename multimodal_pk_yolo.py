import torch
import logging
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class DWConv(nn.Module):
    # Depth-wise convolution layer
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)
        nn.init.kaiming_normal_(self.dwconv.weight, mode='fan_out', nonlinearity='relu')
        if self.dwconv.bias is not None:
            nn.init.zeros_(self.dwconv.bias)

    def forward(self, x):
        return self.dwconv(x)

class RepViTBlock(nn.Module):
    # RepViT block with SE attention 
    # Ao Wang et al., “RepViT: Revisiting Mobile CNN From ViT Perspective”, CVPR 2024
    def __init__(self, inp, oup, stride=1, expand_ratio=4):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        
        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        self.dwconv = DWConv(hidden_dim, stride=stride)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        self.conv2 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)
        
        # Jie Hu et al., “Squeeze-and-Excitation Networks”, CVPR 2018
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(oup, max(1, oup // 16), 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(max(1, oup // 16), oup, 1),
            nn.Sigmoid()
        )
        
        self.act = nn.ReLU6(inplace=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.identity:
            identity = x
            
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.dwconv(out)))
        out = self.bn3(self.conv2(out))
        
        # SE attention
        se_weight = self.se(out)
        out = out * se_weight
        
        if self.identity:
            out = out + identity
            
        return out

class ChannelAttentionBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        for m in self.fc.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        context = self.global_pool(x)
        attention = self.fc(context)
        return x * attention

class Backbone(nn.Module):

    def __init__(self, input_channels=4, width_mult=1.0):
        super().__init__()
        
        self.cfgs = [
            [32, 1, 1],   # Stage 1
            [64, 2, 2],   # Stage 2
            [128, 2, 2],  # Stage 3
            [256, 2, 2],  # Stage 4
            [512, 1, 2],  # Stage 5
        ]
        
        input_channel = int(32 * width_mult)
        
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=False)
        )
        
        self.stages = nn.ModuleList()
        
        for c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            stage_blocks = nn.ModuleList()
            
            for i in range(n):
                stride = s if i == 0 else 1
                stage_blocks.append(RepViTBlock(input_channel, output_channel, stride))
                input_channel = output_channel
            
            self.stages.append(stage_blocks)
        
        # contribution: cross-modal attention (4 modalities)
        self.cross_modal_attention = ChannelAttentionBlock(input_channel)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        
        feature_maps = []
        sizes = []
        for stage_idx, stage_blocks in enumerate(self.stages):
            for block in stage_blocks:
                x = block(x)
            
            if stage_idx >= 1:
                feature_maps.append(x)
                sizes.append(tuple(x.shape[-2:]))
        
        try:
            import logging
            if not hasattr(self, '_sizes_logged'):
                # logging.getLogger(__name__).info(f"[Backbone] feature map sizes (H,W) per stage >=1: {sizes}")
                self._sizes_logged = True
        except Exception:
            pass
        
        # Apply attention to final feature
        if len(feature_maps) > 0:
            feature_maps[-1] = self.cross_modal_attention(feature_maps[-1])
        
        return feature_maps

class FPN(nn.Module):
    #Feature Pyramid Network
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=False)
                )
            )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features):
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(features[i]))
        
        for i in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(
                laterals[i + 1], 
                size=laterals[i].shape[-2:], 
                mode='nearest'
            )
            laterals[i] = laterals[i] + upsampled
        
        fpn_outs = []
        for lateral, fpn_conv in zip(laterals, self.fpn_convs):
            fpn_outs.append(fpn_conv(lateral))
        
        return fpn_outs

class YOLOHead(nn.Module):
    def __init__(self, num_classes=1, in_channels=256, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False)
        )
        
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        self.box_head = nn.Conv2d(in_channels, num_anchors * 4, 1)
        self.obj_head = nn.Conv2d(in_channels, num_anchors, 1)
        
        self._initialize_head()

    def _initialize_head(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        nn.init.constant_(self.obj_head.bias, -2.0)

    def forward(self, x):
        x = self.shared_conv(x)
        
        cls_score = self.cls_head(x)
        bbox_pred = self.box_head(x)
        objectness = self.obj_head(x)
        
        return cls_score, bbox_pred, objectness

class MultimodalPKYOLO(nn.Module):
    def __init__(self, num_classes=1, input_channels=4):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        self.backbone = Backbone(input_channels=input_channels)
        backbone_channels = [64, 128, 256, 512]
        
        self.neck = FPN(backbone_channels, out_channels=256)
        self.head = YOLOHead(num_classes=num_classes, in_channels=256)
        
        self.anchors = torch.tensor([
            [[10, 13], [16, 30], [33, 23]],      # P3/8 - small tumors
            [[30, 61], [62, 45], [59, 119]],     # P4/16 - medium tumors  
            [[116, 90], [156, 198], [373, 326]]  # P5/32 - large tumors
        ], dtype=torch.float32)
        
        self.hyp = {
            'box': 0.05,
            'cls': 0.5,
            'obj': 1.0,
            'anchor_t': 4.0,
            'fl_gamma': 0.0,
            'cls_pw': 1.0,
            'obj_pw': 1.0,
            'label_smoothing': 0.0,
        }

    def forward(self, x):
        features = self.backbone(x)
        fpn_features = self.neck(features)
        fpn_features = fpn_features[-3:]  # P3, P4, P5

        try:
            import logging
            if not hasattr(self, '_fpn_logged'):
                # logging.getLogger(__name__).info(
                #     f"[FPN] output shapes: {[tuple(f.shape[-2:]) for f in fpn_features]}"
                # )
                self._fpn_logged = True
        except Exception:
            pass

        predictions = []
        for feat in fpn_features:
            cls_score, bbox_pred, objectness = self.head(feat)
            predictions.append((cls_score, bbox_pred, objectness))
        return predictions

def create_model(num_classes=1, input_channels=4, pretrained_path=None, device='cuda'):
    model = MultimodalPKYOLO(num_classes=num_classes, input_channels=input_channels)
    
    if pretrained_path and Path(pretrained_path).exists():
        checkpoint = torch.load(pretrained_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded pretrained weights from {pretrained_path}")
    
    return model.to(device)