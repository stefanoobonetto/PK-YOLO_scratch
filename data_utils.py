import os
import shutil
import nibabel as nib
import numpy as np
import cv2
from pathlib import Path
import json
import argparse
from typing import List, Tuple, Dict, Optional
import logging
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BraTSDataProcessor:
    def __init__(self, 
                 input_dir: str, 
                 output_dir: str,
                 img_size: int = 640,
                 slice_range: Tuple[int, int] = (50, 130),
                 pre_split: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.slice_range = slice_range
        self.pre_split = pre_split
        
        if self.pre_split:
            input_splits = [d.name for d in self.input_dir.iterdir() if d.is_dir()]
            if 'val' not in input_splits and 'test' in input_splits:
                splits = ['train', 'val', 'test']
            else:
                splits = input_splits
        else:
            splits = ['train', 'val', 'test']
            
        for split in splits:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def normalize_image(self, img: np.ndarray) -> np.ndarray:
        if img.max() == img.min():
            return np.zeros_like(img, dtype=np.uint8)
        
        img = (img - img.min()) / (img.max() - img.min())
        return (img * 255).astype(np.uint8)
    
    def extract_tumor_bbox(self, seg_slice: np.ndarray) -> List[List[float]]:
        bboxes = []
        
        tumor_mask = seg_slice > 0
        
        if not tumor_mask.any():
            return bboxes
        
        contours, _ = cv2.findContours(
            tumor_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        h, w = seg_slice.shape
        
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue
                
            x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
            
            x_center = (x + bbox_w / 2) / w
            y_center = (y + bbox_h / 2) / h
            norm_width = bbox_w / w
            norm_height = bbox_h / h
            
            bboxes.append([0, x_center, y_center, norm_width, norm_height])
        
        return bboxes
    
    def process_patient(self, patient_dir: Path, split: str) -> int:
        patient_id = patient_dir.name
        
        modalities = ['t1', 't1ce', 't2', 'flair']
        images = {}
        
        for modality in modalities:
            img_file = patient_dir / f"{patient_id}_{modality}.nii.gz"
            if img_file.exists():
                nii = nib.load(str(img_file))
                images[modality] = nii.get_fdata()
            else:
                logger.warning(f"Missing {modality} for {patient_id}")
                return 0
        
        seg_file = patient_dir / f"{patient_id}_seg.nii.gz"
        if seg_file.exists():
            seg_nii = nib.load(str(seg_file))
            segmentation = seg_nii.get_fdata()
        else:
            logger.warning(f"Missing segmentation for {patient_id}")
            return 0
        
        _, _, num_slices = images['t1'].shape
        slices_processed = 0
        
        start_slice = max(0, self.slice_range[0])
        end_slice = min(num_slices, self.slice_range[1])
        
        for slice_idx in range(start_slice, end_slice):
            slice_images = {}
            has_tumor = False
            
            for modality in modalities:
                img_slice = images[modality][:, :, slice_idx]
                img_slice = self.normalize_image(img_slice)
                
                img_slice = cv2.resize(img_slice, (self.img_size, self.img_size))
                slice_images[modality] = img_slice
            
            seg_slice = segmentation[:, :, slice_idx]
            seg_slice = cv2.resize(seg_slice, (self.img_size, self.img_size), 
                                 interpolation=cv2.INTER_NEAREST)
            
            bboxes = self.extract_tumor_bbox(seg_slice)
            
            slice_name = f"{patient_id}_slice_{slice_idx:03d}"
            
            for modality in modalities:
                img_path = self.output_dir / split / 'images' / f"{slice_name}_{modality}.png"
                cv2.imwrite(str(img_path), slice_images[modality])
            
            label_path = self.output_dir / split / 'labels' / f"{slice_name}.txt"
            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    f.write(' '.join(map(str, bbox)) + '\n')
            
            slices_processed += 1
        
        return slices_processed
    
    def process_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        if self.pre_split:
            return self.process_presplit_dataset()
        else:
            return self.process_raw_dataset(train_ratio, val_ratio)
    
    def process_presplit_dataset(self):
        logger.info("Processing pre-split dataset...")
        
        total_stats = {}
        
        available_splits = [d.name for d in self.input_dir.iterdir() if d.is_dir()]
        logger.info(f"Found splits: {available_splits}")
        
        for split in available_splits:
            split_input_dir = self.input_dir / split
            if not split_input_dir.exists():
                logger.warning(f"Split directory {split} not found, skipping...")
                continue
                
            logger.info(f"Processing {split} split...")
            
            patient_dirs = [d for d in split_input_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('BraTS')]
            
            logger.info(f"Found {len(patient_dirs)} patients in {split} split")
            
            total_slices = 0
            for patient_dir in tqdm(patient_dirs, desc=f"Processing {split}"):
                slices_count = self.process_patient(patient_dir, split)
                total_slices += slices_count
            
            total_stats[split] = {
                'patients': len(patient_dirs),
                'slices': total_slices
            }
            
            logger.info(f"{split}: {len(patient_dirs)} patients, {total_slices} slices")
        
        if 'val' not in available_splits and 'train' in available_splits:
            logger.info("Creating validation split from training data...")
            self.create_val_from_train()
            total_stats = self.update_stats_after_val_split(total_stats)
        
        stats_file = self.output_dir / 'dataset_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(total_stats, f, indent=2)
        
        logger.info(f"Dataset processing complete! Stats saved to {stats_file}")
        return total_stats
    
    def process_raw_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        logger.info("Processing raw dataset with automatic splitting...")
        
        patient_dirs = [d for d in self.input_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('BraTS')]
        
        logger.info(f"Found {len(patient_dirs)} patients")
        
        np.random.seed(42)
        np.random.shuffle(patient_dirs)
        
        n_train = int(len(patient_dirs) * train_ratio)
        n_val = int(len(patient_dirs) * val_ratio)
        
        train_patients = patient_dirs[:n_train]
        val_patients = patient_dirs[n_train:n_train + n_val]
        test_patients = patient_dirs[n_train + n_val:]
        
        logger.info(f"Split: {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test")
        
        splits = {
            'train': train_patients,
            'val': val_patients,
            'test': test_patients
        }
        
        total_stats = {}
        
        for split_name, patients in splits.items():
            logger.info(f"Processing {split_name} split...")
            total_slices = 0
            
            for patient_dir in tqdm(patients, desc=f"Processing {split_name}"):
                slices_count = self.process_patient(patient_dir, split_name)
                total_slices += slices_count
            
            total_stats[split_name] = {
                'patients': len(patients),
                'slices': total_slices
            }
            
            logger.info(f"{split_name}: {len(patients)} patients, {total_slices} slices")
        
        stats_file = self.output_dir / 'dataset_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(total_stats, f, indent=2)
        
        logger.info(f"Dataset processing complete! Stats saved to {stats_file}")
        return total_stats
    
    def create_val_from_train(self, val_ratio: float = 0.2):
        train_output_dir = self.output_dir / 'train'
        val_output_dir = self.output_dir / 'val'
        
        train_images = list((train_output_dir / 'images').glob('*_t1.png'))
        
        if not train_images:
            logger.warning("No training images found for validation split creation")
            return
        
        patient_slices = {}
        for img_path in train_images:
            slice_id = img_path.stem.replace('_t1', '')
            if '_slice_' in slice_id:
                patient_id = slice_id.split('_slice_')[0]
                if patient_id not in patient_slices:
                    patient_slices[patient_id] = []
                patient_slices[patient_id].append(slice_id)
        
        patient_ids = list(patient_slices.keys())
        np.random.seed(42)
        np.random.shuffle(patient_ids)
        
        n_val_patients = max(1, int(len(patient_ids) * val_ratio))
        val_patient_ids = patient_ids[:n_val_patients]
        
        logger.info(f"Moving {n_val_patients} patients ({len(patient_ids) - n_val_patients} remain in train)")
        
        moved_slices = 0
        for patient_id in val_patient_ids:
            for slice_id in patient_slices[patient_id]:
                modalities = ['t1', 't1ce', 't2', 'flair']
                for modality in modalities:
                    src_img = train_output_dir / 'images' / f"{slice_id}_{modality}.png"
                    dst_img = val_output_dir / 'images' / f"{slice_id}_{modality}.png"
                    
                    if src_img.exists():
                        src_img.rename(dst_img)
                
                src_label = train_output_dir / 'labels' / f"{slice_id}.txt"
                dst_label = val_output_dir / 'labels' / f"{slice_id}.txt"
                
                if src_label.exists():
                    src_label.rename(dst_label)
                
                moved_slices += 1
        
        logger.info(f"Moved {moved_slices} slices to validation split")
    
    def update_stats_after_val_split(self, original_stats):
        updated_stats = original_stats.copy()
        
        for split in ['train', 'val']:
            split_dir = self.output_dir / split
            if split_dir.exists():
                image_files = list((split_dir / 'images').glob('*_t1.png'))
                label_files = list((split_dir / 'labels').glob('*.txt'))
                
                patient_ids = set()
                for img_path in image_files:
                    slice_id = img_path.stem.replace('_t1', '')
                    if '_slice_' in slice_id:
                        patient_id = slice_id.split('_slice_')[0]
                        patient_ids.add(patient_id)
                
                updated_stats[split] = {
                    'patients': len(patient_ids),
                    'slices': len(image_files)
                }
        
        return updated_stats

def visualize_multimodal_slice(data_dir: str, slice_id: str, save_path: str = None):
    data_path = Path(data_dir)
    
    modalities = ['t1', 't1ce', 't2', 'flair']
    images = {}
    
    for modality in modalities:
        img_path = data_path / 'images' / f"{slice_id}_{modality}.png"
        if img_path.exists():
            images[modality] = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        else:
            logger.warning(f"Missing {modality} for {slice_id}")
            images[modality] = np.zeros((640, 640), dtype=np.uint8)
    
    label_path = data_path / 'labels' / f"{slice_id}.txt"
    bboxes = []
    
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id, x_center, y_center, width, height = map(float, parts[:5])
                    bboxes.append([x_center, y_center, width, height])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'Multimodal Brain MRI - {slice_id}', fontsize=16)
    
    titles = ['T1', 'T1ce', 'T2', 'FLAIR']
    
    for i, (modality, title) in enumerate(zip(modalities, titles)):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        img = images[modality]
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
        
        h, w = img.shape
        for bbox in bboxes:
            x_center, y_center, width, height = bbox
            
            x_center *= w
            y_center *= h
            width *= w
            height *= h
            
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            
            rect = plt.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    
    plt.show()

def analyze_dataset_statistics(data_dir: str):
    data_path = Path(data_dir)
    stats = {}
    
    for split in ['train', 'val', 'test']:
        split_path = data_path / split
        if not split_path.exists():
            continue
        
        image_files = list((split_path / 'images').glob('*_t1.png'))
        label_files = list((split_path / 'labels').glob('*.txt'))
        
        total_boxes = 0
        box_areas = []
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                boxes_in_file = 0
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, _, _, width, height = map(float, parts[:5])
                        area = width * height
                        box_areas.append(area)
                        boxes_in_file += 1
                total_boxes += boxes_in_file
        
        stats[split] = {
            'num_slices': len(image_files),
            'num_labels': len(label_files),
            'total_boxes': total_boxes,
            'avg_boxes_per_slice': total_boxes / len(label_files) if label_files else 0,
            'avg_box_area': np.mean(box_areas) if box_areas else 0,
            'std_box_area': np.std(box_areas) if box_areas else 0
        }
    
    print("\nDataset Statistics:")
    print("=" * 50)
    
    for split, split_stats in stats.items():
        print(f"\n{split.upper()} Split:")
        print(f"  Number of slices: {split_stats['num_slices']}")
        print(f"  Number of label files: {split_stats['num_labels']}")
        print(f"  Total bounding boxes: {split_stats['total_boxes']}")
        print(f"  Avg boxes per slice: {split_stats['avg_boxes_per_slice']:.2f}")
        print(f"  Avg box area: {split_stats['avg_box_area']:.4f}")
        print(f"  Std box area: {split_stats['std_box_area']:.4f}")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        splits = list(stats.keys())
        slice_counts = [stats[split]['num_slices'] for split in splits]
        
        axes[0, 0].bar(splits, slice_counts, color=['blue', 'orange', 'green'])
        axes[0, 0].set_title('Number of Slices per Split')
        axes[0, 0].set_ylabel('Number of Slices')
        
        box_counts = [stats[split]['total_boxes'] for split in splits]
        
        axes[0, 1].bar(splits, box_counts, color=['blue', 'orange', 'green'])
        axes[0, 1].set_title('Total Bounding Boxes per Split')
        axes[0, 1].set_ylabel('Number of Boxes')
        
        avg_boxes = [stats[split]['avg_boxes_per_slice'] for split in splits]
        
        axes[1, 0].bar(splits, avg_boxes, color=['blue', 'orange', 'green'])
        axes[1, 0].set_title('Average Boxes per Slice')
        axes[1, 0].set_ylabel('Avg Boxes per Slice')
        
        all_box_areas = []
        for split in ['train', 'val', 'test']:
            split_path = data_path / split
            if split_path.exists():
                for label_file in (split_path / 'labels').glob('*.txt'):
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                _, _, _, width, height = map(float, parts[:5])
                                area = width * height
                                all_box_areas.append(area)
        
        if all_box_areas:
            axes[1, 1].hist(all_box_areas, bins=50, alpha=0.7, color='purple')
            axes[1, 1].set_title('Distribution of Bounding Box Areas')
            axes[1, 1].set_xlabel('Normalized Area')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plot_path = data_path / 'dataset_statistics.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nDataset statistics plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"\nWarning: Could not create visualization plots: {e}")
        print("This is usually due to display/GUI issues and doesn't affect the data processing.")
    
    return stats

def create_data_yaml(data_dir: str, num_classes: int = 1):
    data_path = Path(data_dir)
    
    yaml_content = f"""path: {data_path.absolute()}
train: train/images
val: val/images
test: test/images

nc: {num_classes}
names: ['tumor']

img_size: 640
batch_size: 8

modalities: ['t1', 't1ce', 't2', 'flair']
input_channels: 4
"""
    
    yaml_path = data_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"Data configuration saved to {yaml_path}")

def validate_dataset(data_dir: str, sample_size: int = 10):
    data_path = Path(data_dir)
    issues = []
    
    print("Validating dataset...")
    
    for split in ['train', 'val', 'test']:
        split_path = data_path / split
        if not split_path.exists():
            issues.append(f"Missing {split} directory")
            continue
        
        print(f"\nValidating {split} split...")
        
        label_files = list((split_path / 'labels').glob('*.txt'))[:sample_size]
        
        for label_file in tqdm(label_files, desc=f"Checking {split} samples"):
            slice_id = label_file.stem
            
            modalities = ['t1', 't1ce', 't2', 'flair']
            missing_modalities = []
            
            for modality in modalities:
                img_path = split_path / 'images' / f"{slice_id}_{modality}.png"
                if not img_path.exists():
                    missing_modalities.append(modality)
            
            if missing_modalities:
                issues.append(f"Missing modalities for {slice_id}: {missing_modalities}")
            
            try:
                with open(label_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) < 5:
                                issues.append(f"Invalid label format in {label_file}:{line_num}")
                            else:
                                try:
                                    class_id, x, y, w, h = map(float, parts[:5])
                                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                        issues.append(f"Invalid bbox coordinates in {label_file}:{line_num}")
                                except ValueError:
                                    issues.append(f"Non-numeric values in {label_file}:{line_num}")
            except Exception as e:
                issues.append(f"Error reading {label_file}: {e}")
    
    if issues:
        print(f"\nFound {len(issues)} issues:")
        for issue in issues[:20]:
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more issues")
    else:
        print("\nDataset validation passed! No issues found.")
    
    return len(issues) == 0

def convert_brats_to_yolo_format(input_dir: str, output_dir: str, pre_split: bool = False):
    processor = BraTSDataProcessor(input_dir, output_dir, pre_split=pre_split)
    
    stats = processor.process_dataset()
    
    create_data_yaml(output_dir)
    
    analyze_dataset_statistics(output_dir)
    
    validate_dataset(output_dir)
    
    print(f"\nDataset conversion complete! Output saved to: {output_dir}")
    print(f"Statistics: {stats}")
    
    return stats

def convert_presplit_brats(input_dir: str, output_dir: str):
    return convert_brats_to_yolo_format(input_dir, output_dir, pre_split=True)

def detect_dataset_structure(input_dir: str) -> bool:
    input_path = Path(input_dir)
    
    subdirs = [d.name for d in input_path.iterdir() if d.is_dir()]
    
    if 'train' in subdirs or 'test' in subdirs:
        logger.info("Detected pre-split dataset structure")
        return True
    
    brats_dirs = [d for d in input_path.iterdir() 
                  if d.is_dir() and d.name.startswith('BraTS')]
    
    if brats_dirs:
        logger.info("Detected raw BraTS dataset structure")
        return False
    
    logger.warning("Could not detect dataset structure")
    return False

class DataAugmentationAnalyzer:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def visualize_augmentations(self, slice_id: str, num_augmentations: int = 6):
        from multimodal_pk_yolo import BraTSDataset
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        dataset = BraTSDataset(
            str(self.data_dir), 
            split='train',
            img_size=640,
            augment=True
        )
        
        sample_idx = None
        for i, sid in enumerate(dataset.slice_ids):
            if slice_id in sid:
                sample_idx = i
                break
        
        if sample_idx is None:
            print(f"Slice {slice_id} not found in dataset")
            return
        
        original_sample = dataset[sample_idx]
        
        fig, axes = plt.subplots(2, num_augmentations//2, figsize=(20, 8))
        fig.suptitle(f'Data Augmentation Examples - {slice_id}', fontsize=16)
        
        for i in range(num_augmentations):
            augmented_sample = dataset[sample_idx]
            
            row, col = i // (num_augmentations//2), i % (num_augmentations//2)
            ax = axes[row, col]
            
            img = augmented_sample['image'][1].numpy()
            ax.imshow(img, cmap='gray')
            
            bboxes = augmented_sample['bboxes']
            if len(bboxes) > 0:
                h, w = img.shape
                for bbox in bboxes:
                    if bbox.sum() > 0:
                        x_center, y_center, width, height = bbox
                        
                        x_center *= w
                        y_center *= h
                        width *= w
                        height *= h
                        
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        
                        rect = plt.Rectangle(
                            (x1, y1), width, height,
                            linewidth=2, edgecolor='red', facecolor='none'
                        )
                        ax.add_patch(rect)
            
            ax.set_title(f'Augmentation {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='BraTS Dataset Processing Utilities')
    parser.add_argument('--action', choices=['convert', 'convert_presplit', 'analyze', 'validate', 'visualize', 'augment', 'detect'], 
                       required=True, help='Action to perform')
    parser.add_argument('--input_dir', type=str, help='Input directory (for convert actions)')
    parser.add_argument('--data_dir', type=str, help='Data directory (for other actions)')
    parser.add_argument('--slice_id', type=str, help='Slice ID (for visualize/augment actions)')
    parser.add_argument('--output_dir', type=str, help='Output directory (for convert actions)')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for processing')
    parser.add_argument('--slice_range', type=int, nargs=2, default=[50, 130], help='Slice range [start, end]')
    
    args = parser.parse_args()
    
    if args.action == 'detect':
        if not args.input_dir:
            print("Error: --input_dir required for detect action")
            return
        is_presplit = detect_dataset_structure(args.input_dir)
        print(f"Dataset structure: {'Pre-split' if is_presplit else 'Raw BraTS format'}")
        if is_presplit:
            print("Use --action convert_presplit")
        else:
            print("Use --action convert")
    
    elif args.action == 'convert':
        if not args.input_dir or not args.output_dir:
            print("Error: --input_dir and --output_dir required for convert action")
            return
        
        is_presplit = detect_dataset_structure(args.input_dir)
        
        processor = BraTSDataProcessor(
            args.input_dir, 
            args.output_dir,
            img_size=args.img_size,
            slice_range=tuple(args.slice_range),
            pre_split=is_presplit
        )
        stats = processor.process_dataset()
        
        create_data_yaml(args.output_dir)
        analyze_dataset_statistics(args.output_dir)
        validate_dataset(args.output_dir)
        
        print(f"\nConversion complete! Stats: {stats}")
    
    elif args.action == 'convert_presplit':
        if not args.input_dir or not args.output_dir:
            print("Error: --input_dir and --output_dir required for convert_presplit action")
            return
        
        stats = convert_presplit_brats(args.input_dir, args.output_dir)
        print(f"Pre-split conversion complete! Stats: {stats}")
    
    elif args.action == 'analyze':
        if not args.data_dir:
            print("Error: --data_dir required for analyze action")
            return
        analyze_dataset_statistics(args.data_dir)
    
    elif args.action == 'validate':
        if not args.data_dir:
            print("Error: --data_dir required for validate action")
            return
        validate_dataset(args.data_dir)
    
    elif args.action == 'visualize':
        if not args.data_dir or not args.slice_id:
            print("Error: --data_dir and --slice_id required for visualize action")
            return
        visualize_multimodal_slice(args.data_dir + "/train", args.slice_id)
    
    elif args.action == 'augment':
        if not args.data_dir or not args.slice_id:
            print("Error: --data_dir and --slice_id required for augment action")
            return
        analyzer = DataAugmentationAnalyzer(args.data_dir)
        analyzer.visualize_augmentations(args.slice_id)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("BraTS Dataset Processing Utilities")
        print("\nUsage examples:")
        print("1. Detect dataset structure:")
        print("   python data_utils.py --action detect --input_dir /path/to/BraTS2020")
        print("\n2. Convert raw BraTS to YOLO format (auto-detects structure):")
        print("   python data_utils.py --action convert --input_dir /path/to/BraTS2020 --output_dir /path/to/output")
        print("\n3. Convert pre-split BraTS to YOLO format:")
        print("   python data_utils.py --action convert_presplit --input_dir /path/to/BraTS2020_TrainingData --output_dir /path/to/output")
        print("\n4. Analyze dataset statistics:")
        print("   python data_utils.py --action analyze --data_dir /path/to/processed_data")
        print("\n5. Validate dataset:")
        print("   python data_utils.py --action validate --data_dir /path/to/processed_data")
        print("\n6. Visualize a slice:")
        print("   python data_utils.py --action visualize --data_dir /path/to/processed_data --slice_id BraTS20_Training_002_slice_029")
    else:
        main()