
def create_default_config(args=None):
    config = {
        'model': {
            'num_classes': 1,
            'input_channels': 4,
            'img_size': 640,
        },
        'training': {
            'batch_size': 4,
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
            'save_interval': 500,
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