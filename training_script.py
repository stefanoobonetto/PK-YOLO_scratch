import os
import torch
import logging
from pathlib import Path
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

# Minimal environment setup
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from loss import YOLOLoss
from multimodal_pk_yolo import create_model
from brats_dataset import BraTSDataset, collate_fn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class Trainer:
    """Simplified trainer for PK-YOLO."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model
        self.model = create_model(
            input_channels=4,
            use_spark_pretrained=bool(args.spark_backbone_path),
            spark_pretrained_path=args.spark_backbone_path,
            device=self.device.type
        ).to(self.device)
        
        # Loss function with optimized settings
        self.criterion = YOLOLoss(self.model, img_size=args.img_size)
        
        # Optimizer with differential learning rates
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
        
        # Tracking
        self.best_loss = float('inf')
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Device: {self.device}")
    
    def _create_optimizer(self):
        """Create optimizer with differential learning rates."""
        # Separate backbone and head parameters
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
        
        # Different learning rates
        param_groups = [
            {'params': backbone_params, 'lr': self.args.lr * self.args.backbone_lr_mult},
            {'params': head_params, 'lr': self.args.lr}
        ]
        
        return optim.AdamW(param_groups, weight_decay=1e-4)
    
    def create_dataloaders(self):
        """Create training and validation dataloaders."""
        data_dir = Path(self.args.data_dir)
        
        # Datasets
        train_dataset = BraTSDataset(
            data_dir, split='train', 
            img_size=self.args.img_size, 
            augment=True
        )
        
        val_dataset = BraTSDataset(
            data_dir, split='val', 
            img_size=self.args.img_size, 
            augment=False
        )
        
        # Weighted sampling for imbalanced data
        weights = []
        for slice_id in train_dataset.slice_ids:
            label_file = data_dir / 'train' / 'labels' / f'{slice_id}.txt'
            # Weight positive samples more
            if label_file.exists() and label_file.stat().st_size > 0:
                with open(label_file) as f:
                    num_tumors = len(f.readlines())
                    # Higher weight for images with small tumors
                    weights.append(3.0 if num_tumors > 0 else 1.0)
            else:
                weights.append(1.0)
        
        sampler = WeightedRandomSampler(weights, len(weights))
        
        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            sampler=sampler,
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True
        )
        
        logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
        return train_loader, val_loader
    
    def train_epoch(self, loader, epoch):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc=f'Epoch {epoch}/{self.args.epochs}')
        for batch in pbar:
            # Move to device
            images = batch['images'].to(self.device, non_blocking=True)
            targets = {
                'images': images,
                'bboxes': batch['bboxes'].to(self.device, non_blocking=True),
                'labels': batch['labels'].to(self.device, non_blocking=True)
            }
            
            # Forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    preds = self.model(images)
                    loss, components = self.criterion(preds, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self.model(images)
                loss, components = self.criterion(preds, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'box': f'{components[0].item():.4f}',
                'obj': f'{components[1].item():.4f}'
            })
        
        return total_loss / len(loader)
    
    @torch.no_grad()
    def validate(self, loader):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(loader, desc='Validation'):
            images = batch['images'].to(self.device, non_blocking=True)
            targets = {
                'images': images,
                'bboxes': batch['bboxes'].to(self.device, non_blocking=True),
                'labels': batch['labels'].to(self.device, non_blocking=True)
            }
            
            preds = self.model(images)
            loss, _ = self.criterion(preds, targets)
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'use_spark_pretrained': hasattr(self.model, 'use_spark_pretrained'),
            'spark_pretrained_path': self.args.spark_backbone_path
        }
        
        # Save regular checkpoint
        if epoch % 10 == 0:
            path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, path)
        
        # Save best model
        if is_best:
            path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, path)
            logger.info(f'Saved best model (loss: {val_loss:.4f})')
    
    def train(self):
        """Main training loop."""
        train_loader, val_loader = self.create_dataloaders()
        
        for epoch in range(1, self.args.epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Logging
            logger.info(
                f'Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                f'Val Loss: {val_loss:.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}'
            )
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.save_checkpoint(epoch, val_loss, is_best=False)
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f'Training complete. Best loss: {self.best_loss:.4f}')


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--spark_backbone_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone_lr_mult', type=float, default=0.1)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--mixed_precision', action='store_true')
    
    args = parser.parse_args()
    
    logger.info('='*50)
    logger.info('PK-YOLO Training (Optimized for Small Tumors)')
    logger.info('='*50)
    
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()