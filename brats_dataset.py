"""
Fixed BraTS Dataset Class that properly handles empty label files
This ensures negative samples (background images) are correctly processed
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BraTSDataset(Dataset):
    """
    BraTS2020 Dataset for multimodal brain tumor detection
    Properly handles empty label files (negative samples)
    """
    
    def __init__(self, data_dir, split='train', img_size=640, augment=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'
        
        # Get all image files
        self.image_dir = self.data_dir / split / 'images'
        self.label_dir = self.data_dir / split / 'labels'
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.label_dir}")
        
        # Find all unique slice identifiers
        self.slice_ids = self._get_slice_ids()
        
        # Setup augmentations
        self.setup_transforms()
        
        # Count positive vs negative samples for debugging
        self._analyze_dataset()
        
        logger.info(f"Loaded {len(self.slice_ids)} samples for {split} split")

    def _get_slice_ids(self):
        """Extract unique slice identifiers from image filenames"""
        slice_ids = set()
        
        # Look for all PNG files in the images directory
        image_files = list(self.image_dir.glob('*.png'))
        
        logger.info(f"Found {len(image_files)} image files in {self.image_dir}")
        
        # Pattern to match: BraTS20_Training_XXX_slice_YYY_modality.png
        pattern = r'(BraTS20_Training_\d+_slice_\d+)_([tf]\w*|flair)\.png'
        
        for img_file in image_files:
            filename = img_file.name
            
            match = re.match(pattern, filename)
            if match:
                slice_id = match.group(1)  # BraTS20_Training_XXX_slice_YYY
                modality = match.group(2)  # t1, t1ce, t2, flair
                slice_ids.add(slice_id)
            else:
                # Try alternative pattern without "slice_" 
                alt_pattern = r'(BraTS20_Training_\d+_\d+)_([tf]\w*|flair)\.png'
                alt_match = re.match(alt_pattern, filename)
                if alt_match:
                    slice_id = alt_match.group(1)
                    slice_ids.add(slice_id)
                else:
                    logger.warning(f"Filename doesn't match expected pattern: {filename}")
        
        if not slice_ids:
            logger.warning(f"No valid slice IDs found in {self.image_dir}")
            # Show some example filenames for debugging
            example_files = [f.name for f in image_files[:10]]
            logger.info(f"Example filenames: {example_files}")
        
        return sorted(list(slice_ids))

    def _analyze_dataset(self):
        """Analyze dataset to count positive vs negative samples"""
        positive_count = 0
        negative_count = 0
        
        for slice_id in self.slice_ids:
            bboxes, _ = self.load_labels(slice_id)
            if len(bboxes) > 0:
                positive_count += 1
            else:
                negative_count += 1
        
        logger.info(f"Dataset analysis for {self.split}:")
        logger.info(f"  Positive samples (with tumors): {positive_count}")
        logger.info(f"  Negative samples (no tumors): {negative_count}")
        logger.info(f"  Total samples: {len(self.slice_ids)}")
        logger.info(f"  Positive ratio: {positive_count/len(self.slice_ids)*100:.1f}%")

    def setup_transforms(self):
        """Setup image transformations"""
        if self.augment:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.485], 
                    std=[0.229, 0.224, 0.225, 0.229],
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.485], 
                    std=[0.229, 0.224, 0.225, 0.229],
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def load_multimodal_image(self, slice_id):
        """Load all 4 modalities for a given slice"""
        modalities = ['t1', 't1ce', 't2', 'flair']
        images = []
        
        for modality in modalities:
            # Try different naming patterns
            possible_paths = [
                self.image_dir / f"{slice_id}_{modality}.png",
                self.image_dir / f"{slice_id}_{modality}.PNG",
            ]
            
            img = None
            for img_path in possible_paths:
                if img_path.exists():
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    break
            
            if img is None:
                logger.warning(f"Missing modality {modality} for slice {slice_id}")
                img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                # Resize if necessary
                if img.shape[:2] != (self.img_size, self.img_size):
                    img = cv2.resize(img, (self.img_size, self.img_size))
            
            images.append(img)
        
        # Stack to create 4-channel image
        multimodal_img = np.stack(images, axis=-1)  # H, W, 4
        return multimodal_img

    def load_labels(self, slice_id):
        """
        Load YOLO format labels - CRITICAL FIX for empty files
        
        IMPORTANT: For tumor class index 0, negative samples must return 
        empty arrays, NOT arrays filled with zeros!
        """
        # Try different label file patterns
        possible_label_paths = [
            self.label_dir / f"{slice_id}.txt",
            self.label_dir / f"{slice_id}.TXT",
        ]
        
        bboxes = []
        class_labels = []
        
        label_path = None
        for path in possible_label_paths:
            if path.exists():
                label_path = path
                break
        
        if label_path and label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                    
                    # Handle empty files explicitly - CRITICAL for class 0
                    if not content:
                        logger.debug(f"Empty label file (negative sample): {label_path}")
                        # Return empty arrays - NOT zeros which would indicate class 0 (tumor)
                        return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.int64)
                    
                    for line in content.split('\n'):
                        line = line.strip()
                        if line:  # Skip empty lines
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center, y_center, width, height = map(float, parts[1:5])
                                
                                # Validate bbox coordinates
                                if all(0 <= coord <= 1 for coord in [x_center, y_center, width, height]):
                                    bboxes.append([x_center, y_center, width, height])
                                    class_labels.append(class_id)
                                else:
                                    logger.warning(f"Invalid bbox coordinates in {label_path}: {line}")
            except Exception as e:
                logger.error(f"Error reading label file {label_path}: {e}")
        else:
            # No label file found - treat as negative sample
            logger.debug(f"No label file found for {slice_id} - treating as negative sample")
        
        # CRITICAL: Return empty arrays for negative samples, not zeros
        if not bboxes:
            return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.int64)
        
        return np.array(bboxes, dtype=np.float32), np.array(class_labels, dtype=np.int64)

    def __len__(self):
        return len(self.slice_ids)

    def __getitem__(self, idx):
        slice_id = self.slice_ids[idx]
        
        # Load multimodal image
        image = self.load_multimodal_image(slice_id)
        
        # Load labels - this now properly handles empty files
        bboxes, class_labels = self.load_labels(slice_id)
        
        # Debug logging for negative samples
        if len(bboxes) == 0:
            logger.debug(f"Loading negative sample: {slice_id}")
        
        # Apply transformations
        try:
            if len(bboxes) > 0:
                # Positive sample with annotations
                transformed = self.transform(
                    image=image, 
                    bboxes=bboxes.tolist(), 
                    class_labels=class_labels.tolist()
                )
                image = transformed['image']
                bboxes = np.array(transformed['bboxes'], dtype=np.float32)
                class_labels = np.array(transformed['class_labels'], dtype=np.int64)
            else:
                # Negative sample (empty label file) - NO objects, NO classes
                transformed = self.transform(image=image, bboxes=[], class_labels=[])
                image = transformed['image']
                # CRITICAL: Keep empty arrays for negative samples
                bboxes = np.zeros((0, 4), dtype=np.float32)
                class_labels = np.array([], dtype=np.int64)  # Empty, not zeros!
                
        except Exception as e:
            logger.error(f"Error applying transforms to {slice_id}: {e}")
            # Fallback: just resize and normalize
            image = cv2.resize(image, (self.img_size, self.img_size))
            image = image.astype(np.float32) / 255.0
            
            # Apply basic normalization
            mean = np.array([0.485, 0.456, 0.406, 0.485])
            std = np.array([0.229, 0.224, 0.225, 0.229])
            for i in range(4):
                image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
            
            # Convert to tensor format (C, H, W)
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            
            # CRITICAL: Empty arrays for negative samples
            bboxes = np.zeros((0, 4), dtype=np.float32)
            class_labels = np.array([], dtype=np.int64)  # Empty, not zeros!
        
        return {
            'image': image,
            'bboxes': bboxes,
            'class_labels': class_labels,
            'slice_id': slice_id
        }

def collate_fn(batch):
    """
    Custom collate function that properly handles negative samples
    CRITICAL: For class index 0, padding must NOT use zeros!
    """
    images = torch.stack([item['image'] for item in batch]).float()
    
    # Handle variable number of bboxes per image
    max_boxes = max(len(item['bboxes']) for item in batch)
    if max_boxes == 0:
        max_boxes = 1  # Ensure we have at least one slot for padding
    
    batch_bboxes = torch.zeros(len(batch), max_boxes, 4, dtype=torch.float32)
    # CRITICAL: Use -1 for padding labels when tumor class is 0
    batch_labels = torch.full((len(batch), max_boxes), -1, dtype=torch.long)
    
    for i, item in enumerate(batch):
        if len(item['bboxes']) > 0:
            num_boxes = len(item['bboxes'])
            batch_bboxes[i, :num_boxes] = torch.from_numpy(item['bboxes']).float()
            batch_labels[i, :num_boxes] = torch.from_numpy(item['class_labels']).long()
        # For negative samples, we leave bboxes as zeros and labels as -1 (padding)
    
    return {
        'images': images,
        'bboxes': batch_bboxes,
        'labels': batch_labels,
        'slice_ids': [item['slice_id'] for item in batch]
    }

def test_dataset(data_dir, split='train'):
    """Test the dataset loading with emphasis on negative samples"""
    logger.info(f"Testing dataset loading for {split} split...")
    
    try:
        dataset = BraTSDataset(data_dir, split=split, img_size=640, augment=False)
        
        if len(dataset) == 0:
            logger.error("Dataset is empty!")
            return False
        
        logger.info(f"Dataset contains {len(dataset)} samples")
        
        # Test loading first few samples to find both positive and negative
        positive_found = False
        negative_found = False
        
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            num_bboxes = len(sample['bboxes'])
            
            if num_bboxes > 0 and not positive_found:
                logger.info(f"Positive sample #{i}: {sample['slice_id']}")
                logger.info(f"  Image shape: {sample['image'].shape}")
                logger.info(f"  Number of bboxes: {num_bboxes}")
                logger.info(f"  Bboxes: {sample['bboxes']}")
                logger.info(f"  Labels: {sample['class_labels']}")
                positive_found = True
            elif num_bboxes == 0 and not negative_found:
                logger.info(f"Negative sample #{i}: {sample['slice_id']}")
                logger.info(f"  Image shape: {sample['image'].shape}")
                logger.info(f"  Number of bboxes: {num_bboxes} (empty - background)")
                negative_found = True
            
            if positive_found and negative_found:
                break
        
        # Test dataloader with both types
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
        
        batch = next(iter(dataloader))
        logger.info(f"Batch test:")
        logger.info(f"  Images shape: {batch['images'].shape}")
        logger.info(f"  Bboxes shape: {batch['bboxes'].shape}")
        logger.info(f"  Labels shape: {batch['labels'].shape}")
        
        # Check for negative samples in batch
        for i in range(len(batch['slice_ids'])):
            has_objects = (batch['bboxes'][i].sum() > 0).item()
            logger.info(f"  Sample {i} ({batch['slice_ids'][i]}): {'Positive' if has_objects else 'Negative'}")
        
        logger.info("✅ Dataset test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test the dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    args = parser.parse_args()
    
    test_dataset(args.data_dir, args.split)