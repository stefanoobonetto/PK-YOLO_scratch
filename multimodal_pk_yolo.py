import torch
import logging
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DWConv(nn.Module):
    """Depth-wise convolution layer."""
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)
        nn.init.kaiming_normal_(self.dwconv.weight, mode='fan_out', nonlinearity='relu')
        if self.dwconv.bias is not None:
            nn.init.zeros_(self.dwconv.bias)

    def forward(self, x):
        return self.dwconv(x)


class RepViTBlock(nn.Module):
    """RepViT block with SE attention."""
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
        
        # SE attention
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
        
        se_weight = self.se(out)
        out = out * se_weight
        
        if self.identity:
            out = out + identity
            
        return out


class ChannelAttentionBlock(nn.Module):
    """Channel attention for cross-modal fusion."""
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


class SparKRepViTBackbone(nn.Module):
    """RepViT backbone compatible with SparK pretrained weights."""
    
    def __init__(self, input_channels=4, width_mult=1.0, spark_pretrained_path=None):
        super().__init__()
        
        self.cfgs = [
            [32, 1, 1],
            [64, 2, 2],
            [128, 2, 2],
            [256, 2, 2],
            [512, 1, 2],
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
        
        self.cross_modal_attention = ChannelAttentionBlock(input_channel)
        
        if spark_pretrained_path and Path(spark_pretrained_path).exists():
            self._load_spark_weights(spark_pretrained_path)
            logger.info(f"Loaded SparK pretrained weights from {spark_pretrained_path}")
        
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _load_spark_weights(self, spark_path):
        """Load SparK pretrained weights."""
        try:
            checkpoint = torch.load(spark_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Extract encoder weights
            encoder_weights = {}
            for key, value in state_dict.items():
                if key.startswith('encoder.'):
                    new_key = key[8:]
                    new_key = new_key.replace('.conv.weight', '.weight')
                    new_key = new_key.replace('.conv.bias', '.bias')
                    encoder_weights[new_key] = value
            
            # Load compatible weights
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in encoder_weights.items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            
            logger.info(f"Loaded {len(pretrained_dict)}/{len(encoder_weights)} pretrained weights")
            
        except Exception as e:
            logger.warning(f"Failed to load SparK weights: {e}")

    def forward(self, x):
        x = self.stem(x)
        
        feature_maps = []
        for stage_idx, stage_blocks in enumerate(self.stages):
            for block in stage_blocks:
                x = block(x)
            
            if stage_idx >= 1:
                feature_maps.append(x)
        
        if len(feature_maps) > 0:
            feature_maps[-1] = self.cross_modal_attention(feature_maps[-1])
        
        return feature_maps


class Backbone(nn.Module):
    """Standard RepViT backbone."""
    
    def __init__(self, input_channels=4, width_mult=1.0):
        super().__init__()
        
        self.cfgs = [
            [32, 1, 1],
            [64, 2, 2],
            [128, 2, 2],
            [256, 2, 2],
            [512, 1, 2],
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
        
        self.cross_modal_attention = ChannelAttentionBlock(input_channel)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        
        feature_maps = []
        for stage_idx, stage_blocks in enumerate(self.stages):
            for block in stage_blocks:
                x = block(x)
            
            if stage_idx >= 1:
                feature_maps.append(x)
        
        if len(feature_maps) > 0:
            feature_maps[-1] = self.cross_modal_attention(feature_maps[-1])
        
        return feature_maps


class FPN(nn.Module):
    """Feature Pyramid Network."""
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
        laterals = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, features)]
        
        for i in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(laterals[i + 1], size=laterals[i].shape[-2:], mode='nearest')
            laterals[i] = laterals[i] + upsampled
        
        fpn_outs = [fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals)]
        
        return fpn_outs


class YOLOHead(nn.Module):
    """YOLO detection head for single-class detection."""
    def __init__(self, in_channels=256, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False)
        )
        
        self.box_head = nn.Conv2d(in_channels, num_anchors * 4, 1)
        self.obj_head = nn.Conv2d(in_channels, num_anchors, 1)
        
        self.initialize_head()

    def initialize_head(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        nn.init.constant_(self.obj_head.bias, -2.0)

    def forward(self, x):
        x = self.shared_conv(x)
        bbox_pred = self.box_head(x)
        objectness = self.obj_head(x)
        return bbox_pred, objectness


class MultimodalPKYOLO(nn.Module):
    """Multimodal PK-YOLO for single-class tumor detection."""
    def __init__(self, input_channels=4, anchors=None, use_spark_pretrained=False, spark_pretrained_path=None):
        super().__init__()
        
        self.input_channels = input_channels
        self.use_spark_pretrained = use_spark_pretrained
        
        if use_spark_pretrained:
            self.backbone = SparKRepViTBackbone(
                input_channels=input_channels,
                spark_pretrained_path=spark_pretrained_path
            )
            logger.info("Using SparK pretrained RepViT backbone")
        else:
            self.backbone = Backbone(input_channels=input_channels)
            logger.info("Using standard RepViT backbone")
        
        backbone_channels = [64, 128, 256, 512]
        
        self.neck = FPN(backbone_channels, out_channels=256)
        self.head = YOLOHead(in_channels=256)
        
        self.anchors = anchors

    def forward(self, x):
        features = self.backbone(x)
        fpn_features = self.neck(features)
        fpn_features = fpn_features[-3:]  # P3, P4, P5

        predictions = []
        for feat in fpn_features:
            bbox_pred, objectness = self.head(feat)
            predictions.append((bbox_pred, objectness))
        
        return predictions


def create_model(input_channels=4, pretrained_path=None, device='cuda', 
                use_spark_pretrained=False, spark_pretrained_path=None):
    """Create MultimodalPKYOLO model with optional SparK pretraining."""
    model = MultimodalPKYOLO(
        input_channels=input_channels,
        use_spark_pretrained=use_spark_pretrained,
        spark_pretrained_path=spark_pretrained_path
    )
    
    if pretrained_path and Path(pretrained_path).exists():
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded full model from {pretrained_path}")
            if 'epoch' in checkpoint:
                logger.info(f"Model trained for {checkpoint['epoch']} epochs")
            if 'best_loss' in checkpoint:
                logger.info(f"Best loss: {checkpoint['best_loss']}")
        else:
            model.load_state_dict(checkpoint)
            logger.info(f"Loaded model weights from {pretrained_path}")
    
    return model.to(device)