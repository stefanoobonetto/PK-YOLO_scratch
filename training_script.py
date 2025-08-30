import os
import time
import torch
import logging
import warnings
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
from multiprocessing import freeze_support
try:
    import cv2
    cv2.setNumThreads(0)
except Exception:
    pass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    torch.set_num_threads(1)
except Exception:
    pass

from loss import YOLOLoss
from utils.utils import get_train_arg_parser, is_positive_label
from utils.config import SimpleConfig
from utils.early_stopping import EarlyStopping
from multimodal_pk_yolo import MultimodalPKYOLO
from brats_dataset import BraTSDataset, collate_fn
from torch.utils.data import WeightedRandomSampler
from utils.train_visualizer import Visualizer

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
        self.model = MultimodalPKYOLO(
            num_classes=self.config.get('model.num_classes', 1),
            input_channels=self.config.get('model.input_channels', 4)
        ).float().to(self.device)
                
        self.criterion = YOLOLoss(
            model=self.model,
            num_classes=self.config.get('model.num_classes', 1),
            autobalance=True
        )
        
        optimizer_type = self.config.get('optimizer.type', 'AdamW')
        lr = self.config.get('training.learning_rate', 1e-3)
        weight_decay = self.config.get('training.weight_decay', 1e-4)
        
        if optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        scheduler_type = self.config.get('optimizer.lr_scheduler', 'CosineAnnealingLR')
        num_epochs = self.config.get('training.num_epochs', 100)
        
        if scheduler_type == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=10)
        elif scheduler_type == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        else:
            self.scheduler = None
            
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        self.output_dir = Path(self.config.get('logging.output_dir', 'outputs'))
        self.output_dir.mkdir(exist_ok=True)
        
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        if self.config.get('training.early_stopping', True):
            self.early_stopping = EarlyStopping(
                patience=self.config.get('training.patience', 20),
                min_delta=self.config.get('training.min_delta', 0.001)
            )
        else:
            self.early_stopping = None
        
        self.scaler = None
        if self.config.get('training.mixed_precision', False):
            try:
                self.scaler = torch.cuda.amp.GradScaler()
            except:
                self.scaler = None
        
        self.visualizer = Visualizer(
            output_dir=str(self.output_dir),
            save_interval=self.config.get('visualization.save_interval', 100),
            conf_thresh=self.config.get('visualization.conf_thresh', 0.05)
        )
                
    def load_datasets(self):
        data_dir = self.config.get('data.data_dir', './data')
        img_size = self.config.get('model.img_size', 640)
        batch_size = self.config.get('training.batch_size', 8)
        num_workers = self.config.get('data.num_workers', 4)

        if os.name == 'nt':
            num_workers = 0
        pin_memory = self.config.get('data.pin_memory', True) and (self.device.type == 'cuda')
        
        train_dir = Path(data_dir) / 'train'
        val_dir = Path(data_dir) / 'val'
        test_dir = Path(data_dir) / 'test'
        
        if not train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        
        if val_dir.exists():
            val_split_name = 'val'
        elif test_dir.exists():
            val_split_name = 'test'
        else:
            raise FileNotFoundError(f"Neither val nor test directory found in {data_dir}")
        
        logger.info(f"Using {val_split_name} as validation set")
        
        self.train_dataset = BraTSDataset(
            data_dir, split='train', img_size=img_size,
            augment=self.config.get('augmentation.enabled', True)
        )

        label_paths = [str(Path(train_dir) / "labels" / f"{sid}.txt") for sid in self.train_dataset.slice_ids]

        weights = []
        for p in label_paths:
            pos = is_positive_label(p)
            # more weight on positive samples (e.g. 3x)
            weights.append(3.0 if pos > 0 else 1.0)

        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0)
        )
        
        self.val_dataset = BraTSDataset(
            data_dir, split=val_split_name, img_size=img_size, augment=False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory
        )
        
        logger.info(f"Loaded {len(self.train_dataset)} training samples")
        logger.info(f"Loaded {len(self.val_dataset)} validation samples")

        try:
            logger.info("Sanity-check: loading one training sample...")
            _sample = self.train_dataset[0]
            _img = _sample['image']
            if isinstance(_img, torch.Tensor):
                logger.info(f"Sample image tensor shape: {_img.shape}; bboxes: {len(_sample['bboxes'])}")
            else:
                logger.info("Sample image loaded (non-tensor).")
        except Exception as e:
            logger.error(f"Sanity-check failed while loading a training sample: {e}")
            raise
        
        if len(self.train_dataset) == 0:
            raise ValueError("Training dataset is empty")
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_components = {'box_loss': 0.0, 'obj_loss': 0.0, 'cls_loss': 0.0}
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                images = batch['images'].to(self.device, dtype=torch.float32, non_blocking=True)
                targets = {
                    'bboxes': batch['bboxes'].to(self.device, dtype=torch.float32, non_blocking=True),
                    'labels': batch['labels'].to(self.device, non_blocking=True),
                    'images': images
                }
                
                self.optimizer.zero_grad()
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(images)
                        loss, loss_components = self.criterion(predictions, targets)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    predictions = self.model(images)
                    loss, loss_components = self.criterion(predictions, targets)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.optimizer.step()
                
                total_loss += loss.item()
                total_components['box_loss'] += loss_components[0].item()
                total_components['obj_loss'] += loss_components[1].item()
                total_components['cls_loss'] += loss_components[2].item()
                
                if self.visualizer.should_save(batch_idx):
                    slice_ids = batch.get('slice_ids', [f'batch_{batch_idx}'])
                    self.visualizer.save_visualization(
                        batch_idx, self.current_epoch, images, targets, predictions, slice_ids
                    )
                
                if batch_idx % 10 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Box': f'{loss_components[0].item():.4f}',
                        'Obj': f'{loss_components[1].item():.4f}',
                        'Cls': f'{loss_components[2].item():.4f}',
                        'LR': f'{current_lr:.6f}'
                    })
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"CUDA OOM at batch {batch_idx}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in total_components.items()}
        
        return avg_loss, avg_components
    
    def validate_epoch(self):
        self.model.eval()
        
        total_loss = 0.0
        total_components = {'box_loss': 0.0, 'obj_loss': 0.0, 'cls_loss': 0.0}
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                try:
                    images = batch['images'].to(self.device, dtype=torch.float32, non_blocking=True)
                    targets = {
                        'bboxes': batch['bboxes'].to(self.device, dtype=torch.float32, non_blocking=True),
                        'labels': batch['labels'].to(self.device, non_blocking=True),
                        'images': images
                    }
                    
                    predictions = self.model(images)
                    loss, loss_components = self.criterion(predictions, targets)
                    
                    total_loss += loss.item()
                    total_components['box_loss'] += loss_components[0].item()
                    total_components['obj_loss'] += loss_components[1].item()
                    total_components['cls_loss'] += loss_components[2].item()
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in total_components.items()}
        
        return avg_loss, avg_components
    
    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
        }
        
        checkpoint_path = self.output_dir / 'checkpoints' / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.output_dir / 'checkpoints' / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def train(self):
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        num_epochs = self.config.get('training.num_epochs', 100)
        save_interval = self.config.get('logging.save_interval', 25)
        
        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                train_loss, _ = self.train_epoch()
                self.train_losses.append(train_loss)
                
                val_loss, _ = self.validate_epoch()
                self.val_losses.append(val_loss)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)
                
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif self.scheduler:
                    self.scheduler.step()
                
                epoch_time = time.time() - epoch_start_time
                
                logger.info(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {epoch_time:.1f}s"
                )
                
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint('best_model.pth', is_best=True)
                    logger.info(f"New best loss: {self.best_loss:.4f}")
                
                if self.early_stopping:
                    if self.early_stopping(val_loss, self.model):
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
                
                if epoch % save_interval == 0 and epoch > 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self.save_checkpoint('final_model.pth')
            logger.info(f"Training completed. Saved {self.visualizer.batch_count} visualizations")



def main():
    parser = get_train_arg_parser()
    args = parser.parse_args()
    
    config = SimpleConfig()
    
    logger.info("======= Multimodal PK-YOLO training - BraTS2020 Dataset =======")
    logger.info(f"  - Data directory: {config.get('data.data_dir')}")
    logger.info(f"  - Output directory: {config.get('logging.output_dir')}")
    logger.info(f"  - Batch size: {config.get('training.batch_size')}")
    logger.info(f"  - Epochs: {config.get('training.num_epochs')}")
    
    try:
        trainer = Trainer(config)
        trainer.load_datasets()
        trainer.train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    freeze_support()
    main()