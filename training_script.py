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

# Robust visualizer import (works whether it's in utils/ or project root)
try:
    from utils.train_visualizer import Visualizer
except ModuleNotFoundError:
    from train_visualizer import Visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class Trainer:
    """Simplified trainer for PK-YOLO."""
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.visualizer = None
        if getattr(args, 'save_visuals', False):
            self.visualizer = Visualizer(
                output_dir=str(self.output_dir),
                save_interval=getattr(args, 'vis_interval', 200),
                conf_thresh=getattr(args, 'vis_conf', 0.5),
            )

        # Model
        self.model = create_model(
            input_channels=4,
            use_spark_pretrained=bool(args.spark_backbone_path),
            spark_pretrained_path=args.spark_backbone_path,
            device=self.device.type
        ).to(self.device)

        # Loss
        self.criterion = YOLOLoss(self.model, img_size=args.img_size)

        # Optimizer with differential LRs
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )

        # Mixed precision — compatible with PyTorch <2.0 and ≥2.0
        if args.mixed_precision:
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                self._use_torch_amp = True
                self.scaler = torch.amp.GradScaler()
            else:
                self._use_torch_amp = False
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self._use_torch_amp = False
            self.scaler = None

        self.best_loss = float('inf')

        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Device: {self.device}")

    def _create_optimizer(self):
        backbone_params, head_params = [], []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                (backbone_params if 'backbone' in name else head_params).append(p)

        param_groups = [
            {'params': backbone_params, 'lr': self.args.lr * self.args.backbone_lr_mult},
            {'params': head_params, 'lr': self.args.lr},
        ]
        return optim.AdamW(param_groups, weight_decay=1e-4)

    def _sanitize_batch(self, batch):
        """
        Ensure images have no NaNs/Infs and bboxes/labels are valid.
        images: [B,4,H,W] float
        bboxes: [B,M,4]  (cx,cy,w,h), normalized or pixels
        labels: [B,M]    ints >= 0
        """
        images = batch['images']
        bboxes = batch['bboxes']
        labels = batch['labels']

        images = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)

        if bboxes.numel() == 0:
            batch['images'] = images
            return batch

        # Normalize if they look like pixels (heuristic)
        if bboxes.max() > 1.5:
            s = float(self.args.img_size)
            bboxes[..., :4] = bboxes[..., :4] / s

        # Clamp and drop degenerate boxes
        bboxes = bboxes.clamp_(0.0, 1.0)
        w, h = bboxes[..., 2], bboxes[..., 3]
        valid = (w > 1e-4) & (h > 1e-4)

        B, M, _ = bboxes.shape
        new_b, new_l = [], []
        for i in range(B):
            vi = valid[i]
            if vi.any():
                new_b.append(bboxes[i][vi])
                new_l.append(labels[i][vi].long())
            else:
                new_b.append(bboxes.new_zeros((0, 4)))
                new_l.append(labels.new_zeros((0,), dtype=torch.long))

        maxm = max(x.shape[0] for x in new_b)
        if maxm == 0:
            b_padded = bboxes.new_zeros((B, 0, 4))
            l_padded = labels.new_zeros((B, 0), dtype=torch.long)
        else:
            b_padded = bboxes.new_zeros((B, maxm, 4))
            l_padded = labels.new_zeros((B, maxm), dtype=torch.long)
            for i in range(B):
                if new_b[i].shape[0] > 0:
                    b_padded[i, :new_b[i].shape[0]] = new_b[i]
                    l_padded[i, :new_l[i].shape[0]] = new_l[i]

        batch['images'] = images
        batch['bboxes'] = b_padded
        batch['labels'] = l_padded
        return batch

    def create_dataloaders(self):
        data_dir = Path(self.args.data_dir)

        train_dataset = BraTSDataset(
            data_dir, split='train', img_size=self.args.img_size, augment=True
        )
        val_dataset = BraTSDataset(
            data_dir, split='val', img_size=self.args.img_size, augment=False
        )

        # Weighted sampling
        weights = []
        for sid in train_dataset.slice_ids:
            lf = data_dir / 'train' / 'labels' / f'{sid}.txt'
            if lf.exists() and lf.stat().st_size > 0:
                with open(lf) as f:
                    n = len(f.readlines())
                    weights.append(3.0 if n > 0 else 1.0)
            else:
                weights.append(1.0)

        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            sampler=sampler,
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=(self.args.workers > 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=(self.args.workers > 0),
        )

        logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
        return train_loader, val_loader

    def _save_vis_safe(self, batch_idx, epoch, images, batch, preds):
        if self.visualizer and self.visualizer.should_save(batch_idx):
            try:
                self.visualizer.save_visualization(
                    batch_idx=batch_idx,
                    epoch=epoch,
                    images=images,
                    targets={'bboxes': batch['bboxes'], 'labels': batch['labels']},
                    predictions=preds,
                    slice_ids=batch.get('slice_ids', []),
                )
            except Exception as e:
                logger.warning(f'Visualizer failed on batch {batch_idx}: {e}')

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}/{self.args.epochs}')
        for batch_idx, batch in pbar:
            batch = self._sanitize_batch(batch)

            images = batch['images'].to(self.device, non_blocking=True)
            targets = {
                'images': images,
                'bboxes': batch['bboxes'].to(self.device, non_blocking=True),
                'labels': batch['labels'].to(self.device, non_blocking=True),
            }

            if self.scaler:
                autocast_ctx = (
                    torch.amp.autocast(device_type="cuda")
                    if self._use_torch_amp else
                    torch.cuda.amp.autocast()
                )
                with autocast_ctx:
                    preds = self.model(images)
                    loss, components = self.criterion(preds, targets)
            else:
                preds = self.model(images)
                loss, components = self.criterion(preds, targets)

            # Non-finite guard (both AMP/FP32)
            if not torch.isfinite(loss):
                logger.warning(
                    f"Non-finite loss at epoch {epoch} batch {batch_idx} | "
                    f"img_nan={torch.isnan(images).any().item()} "
                    f"box_nan={torch.isnan(targets['bboxes']).any().item()} "
                    f"box_min={targets['bboxes'].min().item():.4f} "
                    f"box_max={targets['bboxes'].max().item():.4f}"
                )
                self._save_vis_safe(batch_idx, epoch, images, batch, preds)
                continue

            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()

            # Visuals
            self._save_vis_safe(batch_idx, epoch, images, batch, preds)

            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'box': f'{components[0].item():.4f}',
                'obj': f'{components[1].item():.4f}',
            })

        return total_loss / max(1, len(loader))

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(loader, desc='Validation')):
            batch = self._sanitize_batch(batch)

            images = batch['images'].to(self.device, non_blocking=True)
            targets = {
                'images': images,
                'bboxes': batch['bboxes'].to(self.device, non_blocking=True),
                'labels': batch['labels'].to(self.device, non_blocking=True),
            }

            preds = self.model(images)
            loss, _ = self.criterion(preds, targets)

            if not torch.isfinite(loss):
                logger.warning(f"Non-finite VAL loss at batch {batch_idx}")
                continue

            total_loss += loss.item()

        return total_loss / max(1, len(loader))

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'use_spark_pretrained': hasattr(self.model, 'use_spark_pretrained'),
            'spark_pretrained_path': self.args.spark_backbone_path,
        }

        if epoch % 10 == 0:
            torch.save(ckpt, self.output_dir / f'checkpoint_epoch_{epoch}.pth')

        if is_best:
            torch.save(ckpt, self.output_dir / 'best_model.pth')
            logger.info(f'Saved best model (loss: {val_loss:.4f})')

    def train(self):
        train_loader, val_loader = self.create_dataloaders()

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)

            self.scheduler.step()

            logger.info(
                f'Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                f'Val Loss: {val_loss:.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}'
            )

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.save_checkpoint(epoch, val_loss, is_best=False)

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
    parser.add_argument('--save_visuals', action='store_true', help='Enable saving training visualizations')
    parser.add_argument('--vis_interval', type=int, default=200, help='Save every N batches')
    parser.add_argument('--vis_conf', type=float, default=0.5, help='Confidence threshold for predicted boxes')

    args = parser.parse_args()

    logger.info('='*50)
    logger.info('PK-YOLO Training (Optimized for Small Tumors)')
    logger.info('='*50)

    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
