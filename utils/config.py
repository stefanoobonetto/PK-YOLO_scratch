def create_default_config(args=None):
    config = {
        'model': {
            'num_classes': 1,
            'input_channels': 4,
            'img_size': 640,
            'use_spark_pretrained': False,  # NEW: Enable SparK pretrained backbone
            'spark_pretrained_path': None,  # NEW: Path to SparK pretrained weights
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 100,
            'learning_rate': 0.0001,
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
            'save_interval': 300,
        },
        # NEW: SparK pretraining configuration
        'spark': {
            'enabled': False,
            'pretrain_data_dir': './data/pretrain',  # Directory for SparK pretraining data
            'pretrain_epochs': 300,
            'pretrain_batch_size': 16,
            'pretrain_lr': 1e-4,
            'mask_ratio': 0.75,
            'patch_size': 16,
            'output_dir': './spark_outputs'
        }
    }
    
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
        
        # NEW: SparK-related arguments
        if hasattr(args, 'use_spark_pretrained') and args.use_spark_pretrained:
            config['model']['use_spark_pretrained'] = True
        if hasattr(args, 'spark_pretrained_path') and args.spark_pretrained_path:
            config['model']['spark_pretrained_path'] = args.spark_pretrained_path
    
    return config

class SimpleConfig:
    def __init__(self, config_dict=None):
        self.config = config_dict or create_default_config()
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key, value):
        """Set a configuration value."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value