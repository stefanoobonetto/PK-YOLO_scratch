import os
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
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),
        'input_channels': model.input_channels,
        'num_classes': model.num_classes
    }