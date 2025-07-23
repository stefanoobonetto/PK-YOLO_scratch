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
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--project', type=str, default='multimodal-pk-yolo', help='Wandb project name')
    
    return parser