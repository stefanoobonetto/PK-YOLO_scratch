import os
import torch
import argparse

def get_train_arg_parser():
    parser = argparse.ArgumentParser(description='Train Multimodal PK-YOLO for Brain Tumor Detection')
    
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for logs and checkpoints')
    
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--mixed_precision', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    parser.add_argument('--spark_backbone_path', type=str, help='Path to SparK pretrained backbone weights')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone during training')
    parser.add_argument('--backbone_lr_mult', type=float, default=0.1, help='Learning rate multiplier for backbone')

    return parser

def get_spark_arg_parser():
    """Argument parser specifically for SparK pretraining."""
    parser = argparse.ArgumentParser(description='SparK Pretraining for RepViT Backbone')
    
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Dataset directory for pretraining')
    parser.add_argument('--output_dir', type=str, default='./spark_outputs', 
                       help='Output directory for SparK model')
    parser.add_argument('--epochs', type=int, default=300, help='Number of pretraining epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for pretraining')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Masking ratio for SparK')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size for masking')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    
    return parser

def get_inference_arg_parser():
    """Argument parser for inference with SparK support."""
    parser = argparse.ArgumentParser(description='PK-YOLO Inference with SparK Support')
    
    parser.add_argument('--data_dir', type=str, required=True, help='Root dataset directory')
    parser.add_argument('--weights', type=str, required=True, help='Path to .pth/.pt weights')
    parser.add_argument('--split', type=str, default='val', choices=['train','val','test'])
    parser.add_argument('--save_dir', type=str, default='runs/inference/exp')
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--max_dets', type=int, default=300)
    parser.add_argument('--nms_iou', type=float, default=0.45)
    parser.add_argument('--match_iou', type=float, default=0.50)
    parser.add_argument('--conf_thresh', type=float, default=0.25)
    parser.add_argument('--conf_sweep', type=str, default='')
    parser.add_argument('--best_by', type=str, default='ap50', choices=['f1','ap50','ap5095'])
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--input_channels', type=int, default=4)
    parser.add_argument('--save_vis', action='store_true')
    parser.add_argument('--vis_every', type=int, default=50)
    
    # NEW: SparK support for inference
    parser.add_argument('--use_spark_pretrained', action='store_true',
                       help='Model uses SparK pretrained backbone')
    parser.add_argument('--spark_pretrained_path', type=str,
                       help='Path to SparK pretrained weights (for backbone initialization)')
    
    return parser

def is_positive_label(txt_path):
    if not os.path.exists(txt_path):
        return 0.0
    try:
        return 1.0 if os.path.getsize(txt_path) > 0 else 0.0  
    except:
        return 0.0

def get_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),
        'input_channels': model.input_channels,
        'num_classes': model.num_classes
    }
    
    # Add SparK information if available
    if hasattr(model, 'use_spark_pretrained'):
        info['use_spark_pretrained'] = model.use_spark_pretrained
    
    return info

def build_grid(h: int, w: int, device: torch.device):
    gy, gx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij'
    )
    grid = torch.stack((gx, gy), dim=-1).float()
    return grid

def validate_spark_config(config):
    """Validate SparK configuration parameters."""
    if config.get('model.use_spark_pretrained', False):
        spark_path = config.get('model.spark_pretrained_path')
        if not spark_path:
            raise ValueError("SparK pretrained path must be provided when use_spark_pretrained=True")
        
        from pathlib import Path
        if not Path(spark_path).exists():
            raise FileNotFoundError(f"SparK pretrained weights not found: {spark_path}")
    
    return True