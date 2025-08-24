import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from pathlib import Path
import re
try:
    import cv2 as _cv2
    _cv2.setNumThreads(0)
except Exception:
    pass

logger = logging.getLogger(__name__)

class BraTSDataset(Dataset):
    
    def __init__(self, data_dir, split='train', img_size=640, augment=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'
        
        self.image_dir = self.data_dir / split / 'images'
        self.label_dir = self.data_dir / split / 'labels'
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.label_dir}")
        
        self.slice_ids = self._get_slice_ids()
        self.setup_transforms()
        
        logger.info(f"Loaded {len(self.slice_ids)} samples for {split} split")

    def _get_slice_ids(self):
        # Extract unique slice IDs from filenames
        slice_ids = set()
        image_files = list(self.image_dir.glob('*.png'))
        
        pattern = r'(BraTS20_Training_\d+_slice_\d+)_([tf]\w*|flair)\.png'
        
        for img_file in image_files:
            filename = img_file.name
            match = re.match(pattern, filename)
            if match:
                slice_id = match.group(1)
                slice_ids.add(slice_id)
            else:
                alt_pattern = r'(BraTS20_Training_\d+_\d+)_([tf]\w*|flair)\.png'
                alt_match = re.match(alt_pattern, filename)
                if alt_match:
                    slice_id = alt_match.group(1)
                    slice_ids.add(slice_id)
        
        return sorted(list(slice_ids))

    def setup_transforms(self):
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
        # Load and stack the four modalities
        modalities = ['t1', 't1ce', 't2', 'flair']
        images = []
        
        for modality in modalities:
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
                img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                if img.shape[:2] != (self.img_size, self.img_size):
                    img = cv2.resize(img, (self.img_size, self.img_size))
            
            images.append(img)
        
        multimodal_img = np.stack(images, axis=-1)
        return multimodal_img

    def load_labels(self, slice_id):
        # Load bounding boxes and class labels from label files
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
                    
                    if not content:
                        return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.int64)
                    
                    for line in content.split('\n'):
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center, y_center, width, height = map(float, parts[1:5])
                                
                                if all(0 <= coord <= 1 for coord in [x_center, y_center, width, height]):
                                    bboxes.append([x_center, y_center, width, height])
                                    class_labels.append(class_id)
            except Exception as e:
                logger.error(f"Error reading label file {label_path}: {e}")
        
        if not bboxes:
            return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.int64)
        
        return np.array(bboxes, dtype=np.float32), np.array(class_labels, dtype=np.int64)

    def __len__(self):
        return len(self.slice_ids)

    def __getitem__(self, idx):
        slice_id = self.slice_ids[idx]
        
        image = self.load_multimodal_image(slice_id)
        bboxes, class_labels = self.load_labels(slice_id)
        
        try:
            if len(bboxes) > 0:
                transformed = self.transform(
                    image=image, 
                    bboxes=bboxes.tolist(), 
                    class_labels=class_labels.tolist()
                )
                image = transformed['image']
                bboxes = np.array(transformed['bboxes'], dtype=np.float32)
                class_labels = np.array(transformed['class_labels'], dtype=np.int64)
            else:
                transformed = self.transform(image=image, bboxes=[], class_labels=[])
                image = transformed['image']
                bboxes = np.zeros((0, 4), dtype=np.float32)
                class_labels = np.array([], dtype=np.int64)
                
        except Exception as e:
            logger.error(f"Error applying transforms to {slice_id}: {e}")
            image = cv2.resize(image, (self.img_size, self.img_size))
            image = image.astype(np.float32) / 255.0
            
            mean = np.array([0.485, 0.456, 0.406, 0.485])
            std = np.array([0.229, 0.224, 0.225, 0.229])
            for i in range(4):
                image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
            
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            bboxes = np.zeros((0, 4), dtype=np.float32)
            class_labels = np.array([], dtype=np.int64)
        
        return {
            'image': image,
            'bboxes': bboxes,
            'class_labels': class_labels,
            'slice_id': slice_id
        }

def collate_fn(batch):
    # Custom collate function 
    # fix for variable number of boxes per image
    images = torch.stack([item['image'] for item in batch]).float()
    
    max_boxes = max(len(item['bboxes']) for item in batch)
    if max_boxes == 0:
        max_boxes = 1
    
    batch_bboxes = torch.zeros(len(batch), max_boxes, 4, dtype=torch.float32)
    batch_labels = torch.full((len(batch), max_boxes), -1, dtype=torch.long)
    
    for i, item in enumerate(batch):
        if len(item['bboxes']) > 0:
            num_boxes = len(item['bboxes'])
            batch_bboxes[i, :num_boxes] = torch.from_numpy(item['bboxes']).float()
            batch_labels[i, :num_boxes] = torch.from_numpy(item['class_labels']).long()
    
    return {
        'images': images,
        'bboxes': batch_bboxes,
        'labels': batch_labels,
        'slice_ids': [item['slice_id'] for item in batch]
    }