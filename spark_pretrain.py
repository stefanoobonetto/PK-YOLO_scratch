"""
SparK pretraining implementation for multimodal RepViT backbone.
Optimized for 4-channel MRI data with proper sparse convolution handling.
"""

import torch
import logging
import argparse
import numpy as np
from pathlib import Path
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2

logger = logging.getLogger(__name__)

class SparseConv2d(nn.Module):
    """Sparse convolution that only computes on unmasked patches."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups)
        self.stride = stride
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: input tensor (B, C, H, W)
            mask: binary mask (B, 1, H, W) where 1=visible, 0=masked
        Returns:
            output: convolved features
            output_mask: mask after convolution
        """
        if mask is None:
            return self.conv(x), None
        
        # Apply convolution
        output = self.conv(x * mask)
        
        # Compute output mask - a position is valid if all input positions used are valid
        if self.stride > 1 or any(k > 1 for k in self.kernel_size):
            # Use max pooling to downsample mask properly
            output_mask = F.max_pool2d(
                mask.float(), 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                padding=self.padding
            )
        else:
            output_mask = mask
            
        return output, output_mask

class SparseRepViTBlock(nn.Module):
    """RepViT block adapted for sparse operations with proper mask handling."""
    
    def __init__(self, inp, oup, stride=1, expand_ratio=4):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        
        # Sparse convolutions
        self.conv1 = SparseConv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        self.dwconv = SparseConv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        self.conv2 = SparseConv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)
        
        # SE attention (standard - applied after sparse convs)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(oup, max(1, oup // 16), 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(max(1, oup // 16), oup, 1),
            nn.Sigmoid()
        )
        
        self.act = nn.ReLU6(inplace=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, SparseConv2d)):
                conv_layer = m.conv if hasattr(m, 'conv') else m
                nn.init.kaiming_normal_(conv_layer.weight, mode='fan_out', nonlinearity='relu')
                if conv_layer.bias is not None:
                    nn.init.zeros_(conv_layer.bias)

    def forward(self, x, mask=None):
        identity = x
        identity_mask = mask
        
        # First conv block
        out, mask = self.conv1(x, mask)
        out = self.act(self.bn1(out))
        
        # Depthwise conv
        out, mask = self.dwconv(out, mask)
        out = self.act(self.bn2(out))
        
        # Final conv
        out, mask = self.conv2(out, mask)
        out = self.bn3(out)
        
        # SE attention
        se_weight = self.se(out)
        out = out * se_weight
        
        # Residual connection
        if self.identity:
            out = out + identity
            
        return out, mask

class SparKEncoder(nn.Module):
    """SparK encoder optimized for multimodal MRI."""
    
    def __init__(self, input_channels=4, width_mult=1.0):
        super().__init__()
        
        self.cfgs = [
            [32, 1, 1],   # Stage 1
            [64, 2, 2],   # Stage 2  
            [128, 2, 2],  # Stage 3
            [256, 2, 2],  # Stage 4
            [512, 1, 2],  # Stage 5
        ]
        
        input_channel = int(32 * width_mult)
        
        # Stem
        self.stem = nn.ModuleList([
            SparseConv2d(input_channels, input_channel//2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel//2),
            nn.ReLU6(inplace=False),
            SparseConv2d(input_channel//2, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=False)
        ])
        
        # Stages
        self.stages = nn.ModuleList()
        for c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            stage_blocks = nn.ModuleList()
            
            for i in range(n):
                stride = s if i == 0 else 1
                stage_blocks.append(SparseRepViTBlock(input_channel, output_channel, stride))
                input_channel = output_channel
            
            self.stages.append(stage_blocks)
        
        self.final_channels = input_channel
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, SparseConv2d)):
                conv_layer = m.conv if hasattr(m, 'conv') else m
                nn.init.kaiming_normal_(conv_layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, mask=None):
        # Apply stem
        for i, layer in enumerate(self.stem):
            if isinstance(layer, SparseConv2d):
                x, mask = layer(x, mask)
            else:
                x = layer(x)
        
        # Apply stages
        for stage_blocks in self.stages:
            for block in stage_blocks:
                x, mask = block(x, mask)
        
        return x, mask

class SparKDecoder(nn.Module):
    """Lightweight decoder for SparK reconstruction."""
    
    def __init__(self, encoder_dim=512, output_channels=4):
        super().__init__()
        
        self.decoder_layers = nn.Sequential(
            nn.ConvTranspose2d(encoder_dim, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, 3, 1, 1)
        )
        
    def forward(self, x):
        return self.decoder_layers(x)

class SparKPretrainer(nn.Module):
    """Complete SparK pretraining model."""
    
    def __init__(self, input_channels=4, mask_ratio=0.75):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.encoder = SparKEncoder(input_channels=input_channels)
        self.decoder = SparKDecoder(
            encoder_dim=self.encoder.final_channels,
            output_channels=input_channels
        )
        
    def random_masking(self, x, mask_ratio=None):
        """
        Create random block-wise mask for input.
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        B, C, H, W = x.shape
        
        # Create block-wise mask (16x16 blocks)
        block_size = 16
        num_blocks_h = H // block_size
        num_blocks_w = W // block_size
        total_blocks = num_blocks_h * num_blocks_w
        
        num_masked_blocks = int(mask_ratio * total_blocks)
        
        # Generate mask for each sample
        masks = []
        for b in range(B):
            # Random permutation of block indices
            block_indices = torch.randperm(total_blocks)
            visible_blocks = block_indices[num_masked_blocks:]
            
            # Create spatial mask
            mask = torch.zeros(num_blocks_h, num_blocks_w, device=x.device)
            for idx in visible_blocks:
                i, j = idx // num_blocks_w, idx % num_blocks_w
                mask[i, j] = 1
            
            # Upsample to full resolution
            mask = mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
            masks.append(mask)
        
        # Stack and add channel dimension
        mask = torch.stack(masks, dim=0).unsqueeze(1)  # (B, 1, H, W)
        
        return mask
    
    def forward(self, imgs):
        # Generate mask
        mask = self.random_masking(imgs)
        
        # Encode with mask
        encoded_features, _ = self.encoder(imgs, mask)
        
        # Decode to reconstruct
        reconstruction = self.decoder(encoded_features)
        
        # Ensure reconstruction matches input size
        if reconstruction.shape != imgs.shape:
            reconstruction = F.interpolate(reconstruction, size=imgs.shape[2:], mode='bilinear', align_corners=False)
        
        return reconstruction, mask

class SparKDataset(Dataset):
    """Optimized dataset for SparK pretraining."""
    
    def __init__(self, data_dir, img_size=640, modalities=['t1', 't1ce', 't2', 'flair']):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.modalities = modalities
        
        # Find all available slices
        self.slice_ids = self._find_valid_slices()
        
        logger.info(f"Found {len(self.slice_ids)} complete slices for SparK pretraining")
        
        if len(self.slice_ids) == 0:
            raise ValueError(f"No valid slices found in {data_dir}")
    
    def _find_valid_slices(self):
        """Find slices that have all required modalities."""
        slice_ids = set()
        
        # Try different directory structures
        possible_image_dirs = [
            self.data_dir / 'images',
            self.data_dir / 'train' / 'images',
            self.data_dir / 'val' / 'images',
            self.data_dir / 'test' / 'images',
            self.data_dir
        ]
        
        image_files = []
        for img_dir in possible_image_dirs:
            if img_dir.exists():
                image_files.extend(list(img_dir.glob('*.png')))
                image_files.extend(list(img_dir.glob('*.PNG')))
        
        # Extract slice IDs
        for img_file in image_files:
            filename = img_file.stem
            for mod in self.modalities:
                if filename.endswith(f'_{mod}'):
                    slice_id = filename[:-len(f'_{mod}')]
                    slice_ids.add(slice_id)
                    break
        
        # Verify all modalities exist for each slice
        valid_slices = []
        for slice_id in slice_ids:
            all_exist = True
            for mod in self.modalities:
                # Check all possible locations
                found = False
                for img_dir in possible_image_dirs:
                    if img_dir.exists():
                        possible_paths = [
                            img_dir / f"{slice_id}_{mod}.png",
                            img_dir / f"{slice_id}_{mod}.PNG",
                        ]
                        if any(p.exists() for p in possible_paths):
                            found = True
                            break
                
                if not found:
                    all_exist = False
                    break
            
            if all_exist:
                valid_slices.append(slice_id)
        
        return sorted(valid_slices)
    
    def load_multimodal_image(self, slice_id):
        """Load and preprocess all modalities for a slice."""
        images = []
        
        # Try different directory structures
        possible_image_dirs = [
            self.data_dir / 'images',
            self.data_dir / 'train' / 'images',
            self.data_dir / 'val' / 'images',
            self.data_dir / 'test' / 'images',
            self.data_dir
        ]
        
        for modality in self.modalities:
            img = None
            
            # Search in all possible directories
            for img_dir in possible_image_dirs:
                if img_dir.exists():
                    possible_paths = [
                        img_dir / f"{slice_id}_{modality}.png",
                        img_dir / f"{slice_id}_{modality}.PNG",
                    ]
                    
                    for img_path in possible_paths:
                        if img_path.exists():
                            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                break
                    
                    if img is not None:
                        break
            
            if img is None:
                logger.warning(f"Could not load {modality} for {slice_id}, using zeros")
                img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                # Resize if needed
                if img.shape[:2] != (self.img_size, self.img_size):
                    img = cv2.resize(img, (self.img_size, self.img_size))
            
            images.append(img)
        
        # Stack and normalize
        multimodal_img = np.stack(images, axis=0).astype(np.float32) / 255.0
        
        # Apply normalization
        mean = np.array([0.485, 0.456, 0.406, 0.485])
        std = np.array([0.229, 0.224, 0.225, 0.229])
        
        for i in range(4):
            multimodal_img[i] = (multimodal_img[i] - mean[i]) / std[i]
        
        return multimodal_img
    
    def __len__(self):
        return len(self.slice_ids)
    
    def __getitem__(self, idx):
        slice_id = self.slice_ids[idx]
        image = self.load_multimodal_image(slice_id)
        
        return {
            'image': torch.from_numpy(image).float(),
            'slice_id': slice_id
        }

def train_spark(config):
    """Train SparK pretraining model."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training on device: {device}")
    
    # Model
    model = SparKPretrainer(
        input_channels=4,
        mask_ratio=config.get('mask_ratio', 0.75)
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Dataset
    dataset = SparKDataset(
        data_dir=config['data_dir'],
        img_size=config.get('img_size', 640)
    )
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty - check your data directory structure")
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 0.05)
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get('epochs', 300)
    )
    
    # Training loop
    model.train()
    best_loss = float('inf')
    
    logger.info(f"Starting training for {config.get('epochs', 300)} epochs")
    
    for epoch in range(config.get('epochs', 300)):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstruction, mask = model(images)
            
            # Compute loss (MSE on masked regions)
            loss = F.mse_loss(reconstruction, images, reduction='none')
            
            # Only compute loss on masked regions
            mask_loss = (loss * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
            
            # Backward pass
            mask_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += mask_loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {mask_loss.item():.6f}')
        
        scheduler.step()
        avg_loss = epoch_loss / num_batches
        
        logger.info(f'Epoch {epoch} completed. Average loss: {avg_loss:.6f}')
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, output_dir / 'best_spark_model.pth')
            logger.info(f'New best model saved with loss: {avg_loss:.6f}')
        
        # Regular checkpoint
        if epoch % 50 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, output_dir / f'spark_checkpoint_epoch_{epoch}.pth')

def main():
    parser = argparse.ArgumentParser(description='SparK Pretraining')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='./spark_outputs', help='Output directory')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Masking ratio')
    parser.add_argument('--img_size', type=int, default=640, help='Image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'mask_ratio': args.mask_ratio,
        'img_size': args.img_size,
        'num_workers': args.num_workers,
        'weight_decay': 0.05
    }
    
    logging.basicConfig(level=logging.INFO)
    train_spark(config)

if __name__ == '__main__':
    main()