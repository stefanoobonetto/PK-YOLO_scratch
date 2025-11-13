import yaml
import os
from pathlib import Path


def create_default_config(args=None):
    config = {
        'model': {
            'num_classes': 1,
            'input_channels': 4,
            'img_size': 640,
            'spark_backbone_path': None,
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 100,
            'learning_rate': 0.0001,
            'weight_decay': 0.0001,
            'mixed_precision': True,
            'early_stopping': True,
            'patience': 20,
            'min_delta': 0.001,
            'freeze_backbone': False,
            'backbone_lr_mult': 0.1,
        },
        'data': {
            'data_dir': './data',
            'num_workers': 4,
            'pin_memory': True
        },
        'augmentation': {
            'enabled': True,
        },
        'optimizer': {
            'type': 'AdamW',
            'lr_scheduler': 'CosineAnnealingLR'
        },
        'runtime': {
            'device': 'auto'
        },
        'logging': {
            'output_dir': 'outputs',
            'save_interval': 25
        },
        'visualization': {
            'save_interval': 100,
            'conf_thresh': 0.5
        }
    }
    
    # CLI override
    if args:
        # Data arguments
        if hasattr(args, 'data_dir') and args.data_dir:
            config['data']['data_dir'] = args.data_dir
        
        # Model arguments
        if hasattr(args, 'img_size') and args.img_size:
            config['model']['img_size'] = args.img_size
        if hasattr(args, 'spark_backbone_path') and args.spark_backbone_path:
            config['model']['spark_backbone_path'] = args.spark_backbone_path
        
        # Training arguments
        if hasattr(args, 'batch_size') and args.batch_size:
            config['training']['batch_size'] = args.batch_size
        if hasattr(args, 'epochs') and args.epochs:
            config['training']['num_epochs'] = args.epochs
        if hasattr(args, 'lr') and args.lr:
            config['training']['learning_rate'] = args.lr
        if hasattr(args, 'mixed_precision') and args.mixed_precision:
            config['training']['mixed_precision'] = True
        if hasattr(args, 'early_stopping') and args.early_stopping is not None:
            config['training']['early_stopping'] = args.early_stopping
        if hasattr(args, 'freeze_backbone') and args.freeze_backbone:
            config['training']['freeze_backbone'] = args.freeze_backbone
        if hasattr(args, 'backbone_lr_mult') and args.backbone_lr_mult:
            config['training']['backbone_lr_mult'] = args.backbone_lr_mult
        
        # Runtime arguments
        if hasattr(args, 'device') and args.device:
            config['runtime']['device'] = args.device
        
        # Data loader arguments
        if hasattr(args, 'workers') and args.workers:
            config['data']['num_workers'] = args.workers
        
        # Logging arguments
        if hasattr(args, 'output_dir') and args.output_dir:
            config['logging']['output_dir'] = args.output_dir
    
    # Save config to YAML
    output_dir = Path(config['logging']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Configuration saved to: {config_path}")
    
    return config


class SimpleConfig:
    def __init__(self, config_dict=None):
        self.config = config_dict or create_default_config()
    
    def get(self, key, default=None):
        """Get a configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key, value):
        """Set a configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def to_dict(self):
        """Return the full configuration dictionary."""
        return self.config