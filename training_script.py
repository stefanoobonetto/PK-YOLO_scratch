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
from utils.config import SimpleConfig, create_default_config
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

        # --- Device selection ---
        requested = self.config.get('runtime.device', 'auto')
        if requested == 'cpu':
            self.device = torch.device('cpu')
        elif requested == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                raise RuntimeError("Config requested CUDA but torch.cuda.is_available() is False.")
        else:  # 'auto'
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Optional: cudnn speedups
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        # --- Model with SparK support ---
        self.model = self._create_model_with_spark_support()

        # --- Loss ---
        self.criterion = YOLOLoss(
            model=self.model,
            num_classes=self.config.get('model.num_classes', 1),
            autobalance=True
        )

        # - Focaler‑CIoU come regressione bbox
        self.criterion.hyp.update({
            'reg_metric': 'focaler_ciou',
            'focaler_d': 0.0,   # soglia bassa d
            'focaler_u': 0.95,  # soglia alta u
            'box': 0.05,
            'obj': 1.0,
            'cls': 0.0          # no class loss
        })


        # --- Optimizer with differential learning rates ---
        self.optimizer = self._create_optimizer_with_differential_lr()

        # --- Scheduler ---
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
        self.train_losses, self.val_losses, self.learning_rates = [], [], []

        # --- Output dirs ---
        self.output_dir = Path(self.config.get('logging.output_dir', 'outputs'))
        (self.output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'logs').mkdir(parents=True, exist_ok=True)

        # --- Early stopping ---
        if self.config.get('training.early_stopping', True):
            self.early_stopping = EarlyStopping(
                patience=self.config.get('training.patience', 20),
                min_delta=self.config.get('training.min_delta', 0.001)
            )
        else:
            self.early_stopping = None

        # --- AMP only if CUDA is available and enabled ---
        self.scaler = None
        if self.device.type == 'cuda' and self.config.get('training.mixed_precision', False):
            try:
                self.scaler = torch.cuda.amp.GradScaler()
            except Exception:
                self.scaler = None

        # --- Visualizer ---
        self.visualizer = Visualizer(
            output_dir=str(self.output_dir),
            save_interval=self.config.get('visualization.save_interval', 100),
            conf_thresh=self.config.get('visualization.conf_thresh', 0.5)
        )

        # Log model info
        self._log_model_info()

    def _create_model_with_spark_support(self):
        """Create model with optional SparK pretrained backbone."""
        
        model = MultimodalPKYOLO(
            num_classes=self.config.get('model.num_classes', 1),
            input_channels=self.config.get('model.input_channels', 4)
        ).float().to(self.device)
        
        # Load SparK pretrained backbone if specified
        spark_backbone_path = self.config.get('model.spark_backbone_path', None)
        if spark_backbone_path:
            self._load_spark_backbone_weights(model, spark_backbone_path)
        
        return model
    
    def _load_spark_backbone_weights(self, model, backbone_path):
        """Load SparK pretrained backbone weights."""
        
        backbone_path = Path(backbone_path)
        if not backbone_path.exists():
            raise FileNotFoundError(f"SparK backbone weights not found: {backbone_path}")
        
        logger.info(f"Loading SparK pretrained backbone from: {backbone_path}")
        
        try:
            # Load backbone checkpoint
            checkpoint = torch.load(backbone_path, map_location='cpu')
            
            if 'backbone_state_dict' in checkpoint:
                backbone_weights = checkpoint['backbone_state_dict']
                logger.info(f"Loaded backbone weights extracted from SparK model")
            else:
                # Try to extract from full SparK model
                backbone_weights = self._extract_backbone_from_spark_checkpoint(checkpoint)
            
            # Load weights into model backbone
            model_dict = model.state_dict()
            loaded_weights = {}
            
            for key, value in backbone_weights.items():
                # Map to backbone module
                backbone_key = f'backbone.{key}'
                
                if backbone_key in model_dict:
                    if model_dict[backbone_key].shape == value.shape:
                        loaded_weights[backbone_key] = value
                        logger.debug(f"Loaded: {backbone_key}")
                    else:
                        logger.warning(f"Shape mismatch for {backbone_key}: model={model_dict[backbone_key].shape}, checkpoint={value.shape}")
                else:
                    logger.debug(f"Key not found in model: {backbone_key}")
            
            # Update model state dict
            model_dict.update(loaded_weights)
            model.load_state_dict(model_dict)
            
            logger.info(f"Successfully loaded {len(loaded_weights)} backbone parameters from SparK weights")
            
            # Optionally freeze backbone
            if self.config.get('training.freeze_backbone', False):
                self._freeze_backbone(model)
                logger.info("Backbone frozen - only training detection head")
            
        except Exception as e:
            logger.error(f"Failed to load SparK backbone weights: {e}")
            logger.info("Continuing with random backbone initialization")
    
    def _extract_backbone_from_spark_checkpoint(self, checkpoint):
        """Extract backbone weights from full SparK model checkpoint."""
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        backbone_weights = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                # Remove 'encoder.' prefix and convert sparse conv keys
                new_key = key[8:]
                
                # Convert sparse conv weights to regular conv weights
                if '.conv.weight' in new_key:
                    new_key = new_key.replace('.conv.weight', '.weight')
                elif '.conv.bias' in new_key:
                    new_key = new_key.replace('.conv.bias', '.bias')
                
                backbone_weights[new_key] = value
        
        return backbone_weights
    
    def _freeze_backbone(self, model):
        """Freeze backbone parameters."""
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
    
    def _create_optimizer_with_differential_lr(self):
        """Create optimizer with differential learning rates for backbone vs detection head."""
        
        base_lr = self.config.get('training.learning_rate', 1e-3)
        backbone_lr_mult = self.config.get('training.backbone_lr_mult', 0.1)
        freeze_backbone = self.config.get('training.freeze_backbone', False)
        weight_decay = self.config.get('training.weight_decay', 1e-4)
        optimizer_type = self.config.get('optimizer.type', 'AdamW')
        
        # Separate parameters
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Create parameter groups
        param_groups = []
        
        if backbone_params and not freeze_backbone:
            param_groups.append({
                'params': backbone_params,
                'lr': base_lr * backbone_lr_mult,
                'name': 'backbone'
            })
        
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': base_lr,
                'name': 'detection_head'
            })
        
        # Create optimizer
        if optimizer_type == 'AdamW':
            optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            optimizer = optim.SGD(param_groups, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        return optimizer
    
    def _log_model_info(self):
        """Log detailed model information."""
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        # Count backbone vs detection head parameters
        backbone_params = sum(p.numel() for name, p in self.model.named_parameters() if 'backbone' in name)
        head_params = total_params - backbone_params
        
        logger.info(f"Device: {self.device} | cuda_available={torch.cuda.is_available()} | torch={torch.__version__}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        if frozen_params > 0:
            logger.info(f"Frozen parameters: {frozen_params:,}")
        logger.info(f"Backbone parameters: {backbone_params:,}")
        logger.info(f"Detection head parameters: {head_params:,}")
        
        # Log learning rates
        for i, group in enumerate(self.optimizer.param_groups):
            group_name = group.get('name', f'group_{i}')
            logger.info(f"{group_name} learning rate: {group['lr']:.6f}")
        
        # Log SparK usage
        spark_path = self.config.get('model.spark_backbone_path')
        if spark_path:
            logger.info(f"Using SparK pretrained backbone from: {spark_path}")
        else:
            logger.info("Using randomly initialized backbone")
          
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
            weights.append(3.0 if pos > 0 else 1.0)

        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        
        self.val_dataset = BraTSDataset(
            data_dir, split=val_split_name, img_size=img_size, augment=False
        )
        
        pin_memory = self.config.get('data.pin_memory', True) and (self.device.type == 'cuda')

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

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False
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
            # SparK configuration info
            'spark_backbone_path': self.config.get('model.spark_backbone_path'),
            'freeze_backbone': self.config.get('training.freeze_backbone', False),
            'backbone_lr_mult': self.config.get('training.backbone_lr_mult', 0.1),
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

    # === NEW: flag per abilitare/disabilitare Early Stopping ===
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--early_stopping',
        dest='early_stopping',
        action='store_true',
        help='Abilita l’early stopping (override del config)'
    )
    group.add_argument(
        '--no_early_stopping',
        dest='early_stopping',
        action='store_false',
        help='Disabilita l’early stopping (override del config)'
    )
    parser.set_defaults(early_stopping=None)  # se non passi nulla, non tocchiamo il config

    args = parser.parse_args()

    
    config = SimpleConfig(create_default_config(args))

    # Handle CLI overrides including SparK parameters
    cli_overrides = {
        'data.data_dir': getattr(args, 'data_dir', None),
        'model.img_size': getattr(args, 'img_size', None),
        'training.batch_size': getattr(args, 'batch_size', None),
        'training.num_epochs': getattr(args, 'epochs', None),
        'training.learning_rate': getattr(args, 'lr', None),
        'runtime.device': getattr(args, 'device', None),
        # SparK parameters
        'model.spark_backbone_path': getattr(args, 'spark_backbone_path', None),
        'training.freeze_backbone': getattr(args, 'freeze_backbone', False),
        'training.backbone_lr_mult': getattr(args, 'backbone_lr_mult', 0.1),
        'training.early_stopping': getattr(args, 'early_stopping', None)
    }
    
    for k, v in cli_overrides.items():
        if v is not None:
            config.set(k, v)
    
    logger.info("======= Multimodal PK-YOLO training - BraTS2020 Dataset =======")
    logger.info(f"  - Data directory: {config.get('data.data_dir')}")
    logger.info(f"  - Output directory: {config.get('logging.output_dir')}")
    logger.info(f"  - Batch size: {config.get('training.batch_size')}")
    logger.info(f"  - Epochs: {config.get('training.num_epochs')}")
    logger.info(f"  - Learning rate: {config.get('training.learning_rate')}")
    logger.info(f"  - Device request: {config.get('runtime.device','auto')}")
    logger.info(f"  - Early stopping: {config.get('training.early_stopping', True)}")

    
    # Log SparK configuration
    spark_path = config.get('model.spark_backbone_path')
    if spark_path:
        logger.info(f"  - SparK backbone: {spark_path}")
        logger.info(f"  - Freeze backbone: {config.get('training.freeze_backbone', False)}")
        logger.info(f"  - Backbone LR mult: {config.get('training.backbone_lr_mult', 0.1)}")
    else:
        logger.info("  - Using random backbone initialization")

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