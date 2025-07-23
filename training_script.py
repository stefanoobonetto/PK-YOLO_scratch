import time
import torch
import logging
import warnings
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader

# utils
from complete_yolo_loss import YOLOLoss
from utils.utils import get_train_arg_parser
from utils.early_stopping import EarlyStopping
from multimodal_pk_yolo import MultimodalPKYOLO
from brats_dataset import BraTSDataset, collate_fn
from visualization import DebugVisualizer

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:    
    def __init__(self, config):
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
        # model setup 
        self.model = MultimodalPKYOLO(
            num_classes=self.config.get('model.num_classes', 1),
            input_channels=self.config.get('model.input_channels', 4)
        ).float().to(self.device)
                
        # loss criterion setup
        self.criterion = YOLOLoss(
            model=self.model,
            num_classes=self.config.get('model.num_classes', 1),
            autobalance=True
        )
        
        # optimizer setup
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
        self.best_map = 0.0
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # setup output directories
        self.output_dir = Path(self.config.get('logging.output_dir', 'outputs'))
        self.output_dir.mkdir(exist_ok=True)
        
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        
        # early stopping 
        if self.config.get('training.early_stopping', True):
            self.early_stopping = EarlyStopping(
                patience=self.config.get('training.patience', 20),
                min_delta=self.config.get('training.min_delta', 0.001)
            )
        else:
            self.early_stopping = None
        
        # setup mixed precision
        self.scaler = None
        if self.config.get('training.mixed_precision', False):
            try:
                self.scaler = torch.cuda.amp.GradScaler()
            except:
                self.scaler = None
        
        # Visualization setup
        self.debug_vis = DebugVisualizer(str(self.output_dir), save_interval=100)
    
        
        logger.info(f"🎨 Training visualizer initialized - saving every {self.config.get('visualization.save_interval', 100)} batches")
                
    def load_datasets(self):
        
        data_dir = self.config.get('data.data_dir', './data')
        img_size = self.config.get('model.img_size', 640)
        batch_size = self.config.get('training.batch_size', 8)
        num_workers = self.config.get('data.num_workers', 4)
        pin_memory = self.config.get('data.pin_memory', True)
        
        # Check if data directories exist
        train_dir = Path(data_dir) / 'train'
        val_dir = Path(data_dir) / 'val'
        test_dir = Path(data_dir) / 'test'
        
        if not train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        
        # Use val if exists, otherwise use test
        if val_dir.exists():
            val_split_dir = val_dir
            val_split_name = 'val'
        elif test_dir.exists():
            val_split_dir = test_dir
            val_split_name = 'test'
        else:
            raise FileNotFoundError(f"Neither val nor test directory found in {data_dir}")
        
        logger.info(f"Using {val_split_name} as validation set")
        
        # Training dataset
        self.train_dataset = BraTSDataset(
            data_dir, split='train', img_size=img_size,
            augment=self.config.get('augmentation.enabled', True)
        )
        
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
            drop_last=True  # For stable batch norm
        )
        
        # Validation dataset
        self.val_dataset = BraTSDataset(
            data_dir, split=val_split_name, img_size=img_size, augment=False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory
        )
        
        logger.info(f"Loaded {len(self.train_dataset)} training samples")
        logger.info(f"Loaded {len(self.val_dataset)} validation samples")
        logger.info(f"Training batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
        
        if len(self.train_dataset) == 0:
            raise ValueError("Training dataset is empty! Check your data preprocessing.")
        if len(self.val_dataset) == 0:
            logger.warning("Validation dataset is empty!")
    
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
                
                # mixed precision 
                if self.scaler is not None:
                    # fwd
                    with torch.cuda.amp.autocast():
                        predictions = self.model(images)
                        loss, loss_components = self.criterion(predictions, targets)
                   
                    # bwd
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                # normal w/o mixed precision
                else:
                    # fwd
                    predictions = self.model(images)
                    loss, loss_components = self.criterion(predictions, targets)
                                        
                    self.debug_vis.save_batch_debug(batch_idx, self.current_epoch, images, targets)
                    
                    # bwd 
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.optimizer.step()
                
                total_loss += loss.item()
                total_components['box_loss'] += loss_components[0].item()
                total_components['obj_loss'] += loss_components[1].item()
                total_components['cls_loss'] += loss_components[2].item()
                
                # 🎨 VISUALIZATION: Save training visualization every N batches
                if self.visualizer.should_save(batch_idx):
                    # Get slice IDs from batch
                    slice_ids = batch.get('slice_ids', [f'batch_{batch_idx}_sample_{i}' for i in range(len(images))])
                    
                    logger.info(f"📸 Creating visualization for epoch {self.current_epoch}, batch {batch_idx}")
                    # Save visualization with predictions
                    self.visualizer.update_batch_count(
                        batch_idx=batch_idx,
                        epoch=self.current_epoch,
                        images=images,
                        targets=targets,
                        predictions=predictions,
                        slice_ids=slice_ids
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
                    logger.error(f"CUDA OOM at batch {batch_idx}. Clearing cache and skipping batch.")
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
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
                try:
                    images = batch['images'].to(self.device, dtype=torch.float32, non_blocking=True)
                    targets = {
                        'bboxes': batch['bboxes'].to(self.device, dtype=torch.float32, non_blocking=True),
                        'labels': batch['labels'].to(self.device, non_blocking=True),
                        'images': images
                    }
                    
                    # fwd
                    predictions = self.model(images)
                    loss, loss_components = self.criterion(predictions, targets)
                    
                    total_loss += loss.item()
                    total_components['box_loss'] += loss_components[0].item()
                    total_components['obj_loss'] += loss_components[1].item()
                    total_components['cls_loss'] += loss_components[2].item()
                    
                    # 🎨 VALIDATION VISUALIZATION: Save occasional validation visualizations
                    if batch_idx == 0 and self.current_epoch % 5 == 0:  # First validation batch every 5 epochs
                        slice_ids = batch.get('slice_ids', [f'val_batch_{batch_idx}_sample_{i}' for i in range(len(images))])
                        
                        logger.info(f"📸 Creating validation visualization for epoch {self.current_epoch}")
                        # Save to a different directory for validation
                        val_vis_dir = self.output_dir / 'validation_visualizations'
                        val_vis_dir.mkdir(exist_ok=True)
                        
                        # Create temporary visualizer for validation
                        val_visualizer = TrainingVisualizer(str(self.output_dir), save_interval=1)
                        val_visualizer.vis_dir = val_vis_dir
                        
                        val_visualizer.save_training_visualization(
                            batch_idx=batch_idx,
                            epoch=self.current_epoch,
                            images=images,
                            targets=targets,
                            predictions=predictions,
                            slice_ids=slice_ids
                        )
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning("CUDA OOM during validation. Clearing cache and continuing.")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in total_components.items()}
        
        return avg_loss, avg_components
    
    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_loss': self.best_loss,
            'best_map': self.best_map,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': self.config.config if hasattr(self.config, 'config') else self.config
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
        logger.info(f"🎨 Visualizations will be saved to: {self.output_dir / 'training_visualizations'}")
        
        num_epochs = self.config.get('training.num_epochs', 100)
        save_interval = self.config.get('logging.save_interval', 25)
        
        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # training 
                train_loss, train_components = self.train_epoch()
                self.train_losses.append(train_loss)
                
                # validation 
                val_loss, val_components = self.validate_epoch()
                self.val_losses.append(val_loss)
                
                # update learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)
                
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif self.scheduler:
                    self.scheduler.step()
                
                epoch_time = time.time() - epoch_start_time
                
                # Logging
                logger.info(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {epoch_time:.1f}s"
                )
                
                if epoch % 10 == 0:
                    logger.info(
                        f"  Train Components - Box: {train_components['box_loss']:.4f}, "
                        f"Obj: {train_components['obj_loss']:.4f}, "
                        f"Cls: {train_components['cls_loss']:.4f}"
                    )
                    logger.info(
                        f"  Val Components - Box: {val_components['box_loss']:.4f}, "
                        f"Obj: {val_components['obj_loss']:.4f}, "
                        f"Cls: {val_components['cls_loss']:.4f}"
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
                    logger.info(f"Saved checkpoint at epoch {epoch}")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            self.save_checkpoint('final_model.pth')
            logger.info("Final model saved")
            
            # Log visualization summary
            total_visualizations = self.visualizer.batch_count // self.visualizer.save_interval
            logger.info(f"🎨 Training completed! Created {total_visualizations} training visualizations")
            logger.info(f"📁 Visualizations saved in: {self.output_dir / 'training_visualizations'}")
            
            logger.info("Training completed successfully!")

def create_default_config(args=None):
   """Create default configuration dictionary with optional args override"""
   config = {
       'model': {
           'num_classes': 1,
           'input_channels': 4,
           'img_size': 640,
           'confidence_threshold': 0.25,
           'nms_threshold': 0.45
       },
       'training': {
           'batch_size': 4,  # Reduced for 4-channel input
           'num_epochs': 100,
           'learning_rate': 0.001,
           'weight_decay': 0.0001,
           'mixed_precision': True,
           'early_stopping': True,
           'patience': 20,
           'min_delta': 0.001
       },
       'data': {
           'data_dir': './data',
           'num_workers': 4,
           'pin_memory': True
       },
       'augmentation': {
           'enabled': True,
           'horizontal_flip': 0.5,
           'brightness_contrast': 0.3,
           'gamma': 0.3,
           'noise': 0.3,
           'random_crop_scale': [0.8, 1.0]
       },
       'loss': {
           'ignore_threshold': 0.5,
           'focal_alpha': 0.25,
           'focal_gamma': 2.0
       },
       'optimizer': {
           'type': 'AdamW',
           'lr_scheduler': 'CosineAnnealingLR'
       },
       'logging': {
           'output_dir': 'outputs',
           'save_interval': 25
       },
       'visualization': {
           'save_interval': 100,          # Save visualization every 100 batches
           'modality_for_display': 1,     # 0=T1, 1=T1ce, 2=T2, 3=FLAIR
           'confidence_threshold': 0.1,   # Lower threshold to see more predictions
           'max_predictions_display': 10, # Maximum predictions to show
           'save_validation_vis': True    # Save validation visualizations too
       }
   }
   
   # Override with command line arguments if provided
   if args:
       if args.data_dir:
           config['data']['data_dir'] = args.data_dir
       if args.output_dir:
           config['logging']['output_dir'] = args.output_dir
       if args.batch_size:
           config['training']['batch_size'] = args.batch_size
       if args.epochs:
           config['training']['num_epochs'] = args.epochs
       if args.lr:
           config['training']['learning_rate'] = args.lr
       if args.img_size:
           config['model']['img_size'] = args.img_size
       if args.workers:
           config['data']['num_workers'] = args.workers
       if args.mixed_precision:
           config['training']['mixed_precision'] = True
   
   return config

class SimpleConfig:
    """Simple configuration class"""
    def __init__(self, config_dict=None):
        self.config = config_dict or create_default_config()
    
    def get(self, key, default=None):
        """Get nested configuration value"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

def main():
    """Main function with argument parsing"""
    parser = get_train_arg_parser()
    
    # Add visualization-specific arguments
    parser.add_argument('--vis_interval', type=int, default=20, help='Save visualization every N batches')
    parser.add_argument('--vis_modality', type=int, default=1, choices=[0,1,2,3],
                       help='Modality to use for visualization (0=T1, 1=T1ce, 2=T2, 3=FLAIR)')
    
    args = parser.parse_args()
    
    # config file + override with args input
    config_dict = create_default_config(args)
    
    # Override visualization settings from args
    if args.vis_interval:
        config_dict['visualization']['save_interval'] = args.vis_interval
    if args.vis_modality is not None:
        config_dict['visualization']['modality_for_display'] = args.vis_modality
    
    config = SimpleConfig(config_dict)
    
    logger.info("=" * 70)
    logger.info("🧠 MULTIMODAL PK-YOLO TRAINING WITH VISUALIZATION 🎨")
    logger.info("=" * 70)
    logger.info(f"  Data directory: {config.get('data.data_dir')}")
    logger.info(f"  Output directory: {config.get('logging.output_dir')}")
    logger.info(f"  Batch size: {config.get('training.batch_size')}")
    logger.info(f"  Epochs: {config.get('training.num_epochs')}")
    logger.info(f"  Learning rate: {config.get('training.learning_rate')}")
    logger.info(f"  Image size: {config.get('model.img_size')}")
    logger.info(f"  Mixed precision: {config.get('training.mixed_precision')}")
    logger.info(f"  Early stopping: {config.get('training.early_stopping')}")
    logger.info("  " + "-" * 66)
    logger.info(f"🎨 Visualization interval: every {config.get('visualization.save_interval')} batches")
    logger.info(f"🧠 Display modality: {['T1', 'T1ce', 'T2', 'FLAIR'][config.get('visualization.modality_for_display')]}")
    logger.info(f"🎯 Confidence threshold: {config.get('visualization.confidence_threshold')}")
    logger.info("=" * 70)
    
    try:
        trainer = Trainer(config)
        trainer.load_datasets()
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        return
    
    # Resume from checkpoint if specified
    if args.resume:
        if Path(args.resume).exists():
            # Note: load_checkpoint method would need to be implemented
            logger.info(f"Resume functionality not implemented yet: {args.resume}")
        else:
            logger.error(f"Checkpoint file not found: {args.resume}")
            return
    
    try:
        logger.info("🚀 Starting training process...")
        trainer.train()
        logger.info("✅ [SUCCESS] Training completed successfully!")
        logger.info(f"📊 [RESULTS] Best validation loss: {trainer.best_loss:.4f}")
        logger.info(f"💾 [OUTPUT] Models saved in: {trainer.output_dir / 'checkpoints'}")
        logger.info(f"🎨 [VISUALS] Training visualizations: {trainer.output_dir / 'training_visualizations'}")
        logger.info(f"🔍 [VALIDATION] Validation visualizations: {trainer.output_dir / 'validation_visualizations'}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()