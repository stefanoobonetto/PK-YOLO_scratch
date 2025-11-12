import os
import time
import torch
import logging
import warnings
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib
matplotlib.use('Agg')

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

from loss import YOLOLoss
from utils.utils import get_train_arg_parser, is_positive_label
from utils.config import SimpleConfig, create_default_config
from utils.early_stopping import EarlyStopping
from multimodal_pk_yolo import MultimodalPKYOLO
from brats_dataset import BraTSDataset, collate_fn
from utils.train_visualizer import Visualizer
from multiprocessing import freeze_support

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config, run_output_dir=None):
        self.config = config
        self.device = self.setup_device()

        # Model instantiation
        self.model = self.create_model()
        
        # Loss 
        self.loss_criterion = YOLOLoss(
            model=self.model,
            autobalance=True
        )
        self.loss_criterion.hyp.update({
            'reg_metric': 'focaler_ciou',               # PK-YOLO paper loss
            'focaler_d': 0.0,
            'focaler_u': 0.95,
            'box': 0.05,
            'obj': 1.0
        })
        
        # Optimizer and scheduler
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses, self.val_losses, self.learning_rates = [], [], []
        
        if run_output_dir is not None:
            self.output_dir = Path(run_output_dir)
        else:
            self.output_dir = Path(self.config.get('logging.output_dir', 'outputs'))

        (self.output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        
        # Early stopping
        # if self.config.get('training.early_stopping', True):
        #     self.early_stopping = EarlyStopping(
        #         patience=self.config.get('training.patience', 20),
        #         min_delta=self.config.get('training.min_delta', 0.001)
        #     )
        # else:
        self.early_stopping = None
        
        # Mixed precision
        self.scaler = None
        if self.device.type == 'cuda' and self.config.get('training.mixed_precision', False):
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Visualizer
        self.visualizer = Visualizer(
            output_dir=str(self.output_dir),
            save_interval=self.config.get('visualization.save_interval', 100),
            conf_thresh=self.config.get('visualization.conf_thresh', 0.5)
        )
        
        self.log_model_info()
    
    def setup_device(self):
        """Setup computation device."""
        requested = self.config.get('runtime.device', 'auto')
        
        if requested == 'cpu':
            device = torch.device('cpu')
        elif requested == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            device = torch.device('cuda')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        return device
    
    def create_model(self):
        """Create and initialize model."""
        model = MultimodalPKYOLO(
            input_channels=self.config.get('model.input_channels', 4)
        ).float().to(self.device)
        
        spark_path = self.config.get('model.spark_backbone_path')
        if spark_path:
            self.load_spark_weights(model, spark_path)
        
        return model
    
    def load_spark_weights(self, model, backbone_path):
        """Load pretrained SparK backbone weights."""
        backbone_path = Path(backbone_path)
        if not backbone_path.exists():
            raise FileNotFoundError(f"Backbone weights not found: {backbone_path}")
        
        logger.info(f"Loading SparK backbone from: {backbone_path}")
        
        checkpoint = torch.load(backbone_path, map_location='cpu')
        backbone_weights = checkpoint.get('backbone_state_dict', checkpoint)
        
        # Load weights
        model_dict = model.state_dict()
        loaded = {}
        
        for key, value in backbone_weights.items():
            backbone_key = f'backbone.{key}' if not key.startswith('backbone.') else key
            
            if backbone_key in model_dict and model_dict[backbone_key].shape == value.shape:
                loaded[backbone_key] = value
        
        model_dict.update(loaded)
        model.load_state_dict(model_dict)
        logger.info(f"Loaded {len(loaded)} backbone parameters")
        
        # Freeze if requested
        if self.config.get('training.freeze_backbone', False):
            for name, param in model.named_parameters():
                if 'backbone' in name:
                    param.requires_grad = False
            logger.info("Backbone frozen")
    
    def create_optimizer(self):
        """Differential learning rates for backbone and head."""
        base_lr = self.config.get('training.learning_rate', 1e-3)
        backbone_lr_mult = self.config.get('training.backbone_lr_mult', 0.1)
        weight_decay = self.config.get('training.weight_decay', 1e-4)
        optimizer_type = self.config.get('optimizer.type', 'AdamW')
        
        # Separate backbone and head parameters
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
        
        # Parameter groups
        param_groups = []
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': base_lr * backbone_lr_mult})
        if head_params:
            param_groups.append({'params': head_params, 'lr': base_lr})
        
        # Create optimizer
        if optimizer_type == 'AdamW':
            return optim.AdamW(param_groups, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            return optim.SGD(param_groups, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def create_scheduler(self):
        """Learning rate scheduler."""
        scheduler_type = self.config.get('optimizer.lr_scheduler', 'CosineAnnealingLR')
        num_epochs = self.config.get('training.num_epochs', 100)
        
        if scheduler_type == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif scheduler_type == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=10)
        elif scheduler_type == 'StepLR':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        return None
    
    def log_model_info(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        for i, group in enumerate(self.optimizer.param_groups):
            logger.info(f"Group {i} LR: {group['lr']:.6f}")
    def load_datasets(self):
        data_dir = Path(self.config.get('data.data_dir', './data'))
        img_size = self.config.get('model.img_size', 640)
        batch_size = self.config.get('training.batch_size', 8)
        num_workers = 0 if os.name == 'nt' else self.config.get('data.num_workers', 4)
        pin_memory = self.device.type == 'cuda' and self.config.get('data.pin_memory', True)
        
        train_dir = data_dir / 'train'
        val_dir = data_dir / 'val'
        
        if not train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not val_dir.exists():
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
                
        # Create datasets
        self.train_dataset = BraTSDataset(
            data_dir, 
            # split='train_clean', 
            split='train', 
            img_size=img_size,
            augment=self.config.get('augmentation.enabled', True)
        )
        
        self.val_dataset = BraTSDataset(
            data_dir, 
            # split='val_clean', 
            split='val', 
            img_size=img_size, 
            augment=False
        )
        
        # Create weighted sampler for training
        weights = []
        for sid in self.train_dataset.slice_ids:
            label_path = train_dir / "labels" / f"{sid}.txt"
            pos = is_positive_label(str(label_path))
            weights.append(3.0 if pos > 0 else 1.0)
        
        train_sampler = WeightedRandomSampler(
            weights, 
            num_samples=len(weights), 
            replacement=True
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0)
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0)
        )
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
        
    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_components = {'box_loss': 0.0, 'obj_loss': 0.0}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device, dtype=torch.float32, non_blocking=True)
            targets = {
                'bboxes': batch['bboxes'].to(self.device, dtype=torch.float32, non_blocking=True),
                'labels': batch['labels'].to(self.device, non_blocking=True),
                'images': images
            }
            
            self.optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    predictions = self.model(images)
                    loss, loss_components = self.loss_criterion(predictions, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(images)
                loss, loss_components = self.loss_criterion(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_components['box_loss'] += loss_components[0].item()
            total_components['obj_loss'] += loss_components[1].item()
            
            # Visualization
            if self.visualizer.should_save(batch_idx):
                slice_ids = batch.get('slice_ids', [f'batch_{batch_idx}'])
                self.visualizer.save_visualization(
                    batch_idx, self.current_epoch, images, targets, predictions, slice_ids
                )
            
            # bar update
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Box': f'{loss_components[0].item():.4f}',
                    'Obj': f'{loss_components[1].item():.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        num_batches = len(self.train_loader)
        return total_loss / num_batches, {k: v / num_batches for k, v in total_components.items()}
    
    def validate_one_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_components = {'box_loss': 0.0, 'obj_loss': 0.0}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['images'].to(self.device, dtype=torch.float32, non_blocking=True)
                targets = {
                    'bboxes': batch['bboxes'].to(self.device, dtype=torch.float32, non_blocking=True),
                    'labels': batch['labels'].to(self.device, non_blocking=True),
                    'images': images
                }
                
                predictions = self.model(images)
                loss, loss_components = self.loss_criterion(predictions, targets)
                
                total_loss += loss.item()
                total_components['box_loss'] += loss_components[0].item()
                total_components['obj_loss'] += loss_components[1].item()
        
        num_batches = len(self.val_loader)
        return total_loss / num_batches, {k: v / num_batches for k, v in total_components.items()}
    
    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        """Save training checkpoint."""
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
        
        num_epochs = self.config.get('training.num_epochs', 100)
        save_interval = self.config.get('logging.save_interval', 25)
        
        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                epoch_start = time.time()
                
                # Train and validate
                train_loss, _ = self.train_one_epoch()
                val_loss, _ = self.validate_one_epoch()
                
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                
                # Update learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)
                
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif self.scheduler:
                    self.scheduler.step()
                
                # Log progress
                epoch_time = time.time() - epoch_start
                logger.info(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
                )
                
                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint('best_model.pth', is_best=True)
                
                # Early stopping
                if self.early_stopping and self.early_stopping(val_loss, self.model):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Periodic checkpoint
                if epoch % save_interval == 0 and epoch > 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
                
                # Clean cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted")
        finally:
            self.save_checkpoint('final_model.pth')
            logger.info(f"Training completed. Best loss: {self.best_loss:.4f}")


def main():
    parser = get_train_arg_parser()
    parser.add_argument('--no_early_stopping', dest='early_stopping', action='store_false')
    parser.set_defaults(early_stopping=None)
    args = parser.parse_args()
    
    config = SimpleConfig(create_default_config(args))
    
    effective_out_dir = getattr(args, 'output_dir', None) or config.get('logging.output_dir', 'outputs')
    logger.info(f"Effective output dir (used by Visualizer & checkpoints): {effective_out_dir}")
     
    logger.info("=" * 60)
    logger.info("Multimodal PK-YOLO Training")
    logger.info("=" * 60)
    logger.info(f"Data dir: {config.get('data.data_dir')}")
    logger.info(f"Output dir: {effective_out_dir}")
    logger.info(f"Batch size: {config.get('training.batch_size')}")
    logger.info(f"Epochs: {config.get('training.num_epochs')}")
    logger.info(f"Learning rate: {config.get('training.learning_rate')}")
    logger.info(f"Device: {config.get('runtime.device')}")
    logger.info(f"Early stopping: {config.get('training.early_stopping')}")
    
    spark_path = config.get('model.spark_backbone_path')
    if spark_path:
        logger.info(f"SparK backbone: {spark_path}")
        logger.info(f"Freeze backbone: {config.get('training.freeze_backbone')}")
    
    # Train
    trainer = Trainer(config, run_output_dir=effective_out_dir)
    trainer.load_datasets()
    trainer.train()

if __name__ == "__main__":
    freeze_support()
    main()