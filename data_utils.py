"""
Data utilities and helper functions for Multimodal PK-YOLO
"""

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
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BraTSDataProcessor:
    """
    Process BraTS2020 NIfTI files to PNG slices with YOLO format labels
    """
    
    def __init__(self, 
                 input_dir: str, 
                 output_dir: str,
                 img_size: int = 640,
                 slice_range: Tuple[int, int] = (50, 130)):
        """
        Args:
            input_dir: Directory containing BraTS2020 NIfTI files
            output_dir: Output directory for processed data
            img_size: Target image size
            slice_range: Range of slices to extract (to avoid empty slices)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.slice_range = slice_range
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 range"""
        if img.max() == img.min():
            return np.zeros_like(img, dtype=np.uint8)
        
        img = (img - img.min()) / (img.max() - img.min())
        return (img * 255).astype(np.uint8)
    
    def extract_tumor_bbox(self, seg_slice: np.ndarray) -> List[List[float]]:
        """
        Extract bounding boxes from segmentation mask in YOLO format
        
        Args:
            seg_slice: 2D segmentation mask
            
        Returns:
            List of bounding boxes in YOLO format [class_id, x_center, y_center, width, height]
        """
        bboxes = []
        
        # BraTS labels: 0=background, 1=necrotic, 2=edema, 4=enhancing
        # We'll treat all non-zero as tumor (class 0)
        tumor_mask = seg_slice > 0
        
        if not tumor_mask.any():
            return bboxes
        
        # Find connected components
        contours, _ = cv2.findContours(
            tumor_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        h, w = seg_slice.shape
        
        for contour in contours:
            if cv2.contourArea(contour) < 100:  # Skip small artifacts
                continue
                
            x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
            
            # Convert to YOLO format (normalized)
            x_center = (x + bbox_w / 2) / w
            y_center = (y + bbox_h / 2) / h
            norm_width = bbox_w / w
            norm_height = bbox_h / h
            
            # Class 0 for tumor
            bboxes.append([0, x_center, y_center, norm_width, norm_height])
        
        return bboxes
    
    def process_patient(self, patient_dir: Path, split: str) -> int:
        """
        Process a single patient's data
        
        Args:
            patient_dir: Path to patient directory
            split: Data split (train/val/test)
            
        Returns:
            Number of slices processed
        """
        patient_id = patient_dir.name
        
        # Load NIfTI files
        modalities = ['t1', 't1ce', 't2', 'flair']
        images = {}
        
        # Load images
        for modality in modalities:
            img_file = patient_dir / f"{patient_id}_{modality}.nii.gz"
            if img_file.exists():
                nii = nib.load(str(img_file))
                images[modality] = nii.get_fdata()
            else:
                logger.warning(f"Missing {modality} for {patient_id}")
                return 0
        
        # Load segmentation
        seg_file = patient_dir / f"{patient_id}_seg.nii.gz"
        if seg_file.exists():
            seg_nii = nib.load(str(seg_file))
            segmentation = seg_nii.get_fdata()
        else:
            logger.warning(f"Missing segmentation for {patient_id}")
            return 0
        
        # Get dimensions
        _, _, num_slices = images['t1'].shape
        slices_processed = 0
        
        # Process each slice
        start_slice = max(0, self.slice_range[0])
        end_slice = min(num_slices, self.slice_range[1])
        
        for slice_idx in range(start_slice, end_slice):
            # Extract and process each modality
            slice_images = {}
            has_tumor = False
            
            for modality in modalities:
                img_slice = images[modality][:, :, slice_idx]
                img_slice = self.normalize_image(img_slice)
                
                # Resize to target size
                img_slice = cv2.resize(img_slice, (self.img_size, self.img_size))
                slice_images[modality] = img_slice
            
            # Process segmentation
            seg_slice = segmentation[:, :, slice_idx]
            seg_slice = cv2.resize(seg_slice, (self.img_size, self.img_size), 
                                 interpolation=cv2.INTER_NEAREST)
            
            # Extract bounding boxes
            bboxes = self.extract_tumor_bbox(seg_slice)
            
            # Save slice and labels
            slice_name = f"{patient_id}_slice_{slice_idx:03d}"
            
            # Save images
            for modality in modalities:
                img_path = self.output_dir / split / 'images' / f"{slice_name}_{modality}.png"
                cv2.imwrite(str(img_path), slice_images[modality])
            
            # Save labels
            label_path = self.output_dir / split / 'labels' / f"{slice_name}.txt"
            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    f.write(' '.join(map(str, bbox)) + '\n')
            
            slices_processed += 1
        
        return slices_processed
    
    def process_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """
        Process entire BraTS2020 dataset
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation (remaining goes to test)
        """
        # Get all patient directories
        patient_dirs = [d for d in self.input_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('BraTS20_Training')]
        
        logger.info(f"Found {len(patient_dirs)} patients")
        
        # Split patients
        np.random.seed(42)  # For reproducible splits
        np.random.shuffle(patient_dirs)
        
        n_train = int(len(patient_dirs) * train_ratio)
        n_val = int(len(patient_dirs) * val_ratio)
        
        train_patients = patient_dirs[:n_train]
        val_patients = patient_dirs[n_train:n_train + n_val]
        test_patients = patient_dirs[n_train + n_val:]
        
        logger.info(f"Split: {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test")
        
        # Process each split
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
        
        # Save dataset statistics
        stats_file = self.output_dir / 'dataset_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(total_stats, f, indent=2)
        
        logger.info(f"Dataset processing complete! Stats saved to {stats_file}")

def visualize_multimodal_slice(data_dir: str, slice_id: str, save_path: str = None):
    """
    Visualize a multimodal slice with annotations
    
    Args:
        data_dir: Directory containing processed data
        slice_id: Slice identifier (without modality suffix)
        save_path: Path to save visualization
    """
    data_path = Path(data_dir)
    
    # Load all modalities
    modalities = ['t1', 't1ce', 't2', 'flair']
    images = {}
    
    for modality in modalities:
        img_path = data_path / 'images' / f"{slice_id}_{modality}.png"
        if img_path.exists():
            images[modality] = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        else:
            logger.warning(f"Missing {modality} for {slice_id}")
            images[modality] = np.zeros((640, 640), dtype=np.uint8)
    
    # Load labels
    label_path = data_path / 'labels' / f"{slice_id}.txt"
    bboxes = []
    
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id, x_center, y_center, width, height = map(float, parts[:5])
                    bboxes.append([x_center, y_center, width, height])
    
    # Create visualization
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
        
        # Draw bounding boxes
        h, w = img.shape
        for bbox in bboxes:
            x_center, y_center, width, height = bbox
            
            # Convert to pixel coordinates
            x_center *= w
            y_center *= h
            width *= w
            height *= h
            
            # Convert to corner format
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            
            # Draw rectangle
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
    """
    Analyze and visualize dataset statistics
    
    Args:
        data_dir: Directory containing processed dataset
    """
    data_path = Path(data_dir)
    stats = {}
    
    for split in ['train', 'val', 'test']:
        split_path = data_path / split
        if not split_path.exists():
            continue
        
        # Count images and labels
        image_files = list((split_path / 'images').glob('*_t1.png'))  # Count unique slices
        label_files = list((split_path / 'labels').glob('*.txt'))
        
        # Analyze bounding boxes
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
    
    # Print statistics
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
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Number of slices per split
    splits = list(stats.keys())
    slice_counts = [stats[split]['num_slices'] for split in splits]
    
    axes[0, 0].bar(splits, slice_counts, color=['blue', 'orange', 'green'])
    axes[0, 0].set_title('Number of Slices per Split')
    axes[0, 0].set_ylabel('Number of Slices')
    
    # Plot 2: Total bounding boxes per split
    box_counts = [stats[split]['total_boxes'] for split in splits]
    
    axes[0, 1].bar(splits, box_counts, color=['blue', 'orange', 'green'])
    axes[0, 1].set_title('Total Bounding Boxes per Split')
    axes[0, 1].set_ylabel('Number of Boxes')
    
    # Plot 3: Average boxes per slice
    avg_boxes = [stats[split]['avg_boxes_per_slice'] for split in splits]
    
    axes[1, 0].bar(splits, avg_boxes, color=['blue', 'orange', 'green'])
    axes[1, 0].set_title('Average Boxes per Slice')
    axes[1, 0].set_ylabel('Avg Boxes per Slice')
    
    # Plot 4: Box area distribution (combine all splits)
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
    plt.savefig(data_path / 'dataset_statistics.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return stats

def create_data_yaml(data_dir: str, num_classes: int = 1):
    """
    Create YAML configuration file for training
    
    Args:
        data_dir: Path to dataset directory
        num_classes: Number of classes
    """
    data_path = Path(data_dir)
    
    yaml_content = f"""# Multimodal PK-YOLO Dataset Configuration

# Dataset paths
path: {data_path.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
nc: {num_classes}  # number of classes
names: ['tumor']  # class names

# Image settings
img_size: 640
batch_size: 8

# Modalities
modalities: ['t1', 't1ce', 't2', 'flair']
input_channels: 4
"""
    
    yaml_path = data_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"Data configuration saved to {yaml_path}")

def validate_dataset(data_dir: str, sample_size: int = 10):
    """
    Validate dataset integrity and show sample data
    
    Args:
        data_dir: Directory containing processed dataset
        sample_size: Number of samples to validate
    """
    data_path = Path(data_dir)
    issues = []
    
    print("Validating dataset...")
    
    for split in ['train', 'val', 'test']:
        split_path = data_path / split
        if not split_path.exists():
            issues.append(f"Missing {split} directory")
            continue
        
        print(f"\nValidating {split} split...")
        
        # Get sample of label files
        label_files = list((split_path / 'labels').glob('*.txt'))[:sample_size]
        
        for label_file in tqdm(label_files, desc=f"Checking {split} samples"):
            slice_id = label_file.stem
            
            # Check if all modality images exist
            modalities = ['t1', 't1ce', 't2', 'flair']
            missing_modalities = []
            
            for modality in modalities:
                img_path = split_path / 'images' / f"{slice_id}_{modality}.png"
                if not img_path.exists():
                    missing_modalities.append(modality)
            
            if missing_modalities:
                issues.append(f"Missing modalities for {slice_id}: {missing_modalities}")
            
            # Validate label format
            try:
                with open(label_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():  # Skip empty lines
                            parts = line.strip().split()
                            if len(parts) < 5:
                                issues.append(f"Invalid label format in {label_file}:{line_num}")
                            else:
                                # Check if values are valid
                                try:
                                    class_id, x, y, w, h = map(float, parts[:5])
                                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                        issues.append(f"Invalid bbox coordinates in {label_file}:{line_num}")
                                except ValueError:
                                    issues.append(f"Non-numeric values in {label_file}:{line_num}")
            except Exception as e:
                issues.append(f"Error reading {label_file}: {e}")
    
    # Print validation results
    if issues:
        print(f"\nFound {len(issues)} issues:")
        for issue in issues[:20]:  # Show first 20 issues
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more issues")
    else:
        print("\nDataset validation passed! No issues found.")
    
    return len(issues) == 0

def convert_brats_to_yolo_format(input_dir: str, output_dir: str):
    """
    Main function to convert BraTS dataset to YOLO format
    
    Args:
        input_dir: Directory containing original BraTS NIfTI files
        output_dir: Output directory for processed dataset
    """
    processor = BraTSDataProcessor(input_dir, output_dir)
    
    # Process the dataset
    processor.process_dataset()
    
    # Create data configuration
    create_data_yaml(output_dir)
    
    # Analyze statistics
    analyze_dataset_statistics(output_dir)
    
    # Validate dataset
    validate_dataset(output_dir)
    
    print(f"\nDataset conversion complete! Output saved to: {output_dir}")

class DataAugmentationAnalyzer:
    """Analyze the effect of data augmentations"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def visualize_augmentations(self, slice_id: str, num_augmentations: int = 6):
        """
        Visualize the effect of different augmentations on a sample
        
        Args:
            slice_id: Slice identifier to augment
            num_augmentations: Number of augmentation examples to show
        """
        from multimodal_pk_yolo import BraTSDataset
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        # Create dataset with augmentations
        dataset = BraTSDataset(
            str(self.data_dir), 
            split='train',
            img_size=640,
            augment=True
        )
        
        # Find the sample
        sample_idx = None
        for i, sid in enumerate(dataset.slice_ids):
            if slice_id in sid:
                sample_idx = i
                break
        
        if sample_idx is None:
            print(f"Slice {slice_id} not found in dataset")
            return
        
        # Load original sample
        original_sample = dataset[sample_idx]
        
        # Create figure
        fig, axes = plt.subplots(2, num_augmentations//2, figsize=(20, 8))
        fig.suptitle(f'Data Augmentation Examples - {slice_id}', fontsize=16)
        
        for i in range(num_augmentations):
            # Get augmented sample
            augmented_sample = dataset[sample_idx]
            
            row, col = i // (num_augmentations//2), i % (num_augmentations//2)
            ax = axes[row, col]
            
            # Show T1ce modality (channel 1)
            img = augmented_sample['image'][1].numpy()  # T1ce channel
            ax.imshow(img, cmap='gray')
            
            # Draw bounding boxes
            bboxes = augmented_sample['bboxes']
            if len(bboxes) > 0:
                h, w = img.shape
                for bbox in bboxes:
                    if bbox.sum() > 0:  # Valid bbox
                        x_center, y_center, width, height = bbox
                        
                        # Convert to pixel coordinates
                        x_center *= w
                        y_center *= h
                        width *= w
                        height *= h
                        
                        # Convert to corner format
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        
                        # Draw rectangle
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
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='BraTS Dataset Processing Utilities')
    parser.add_argument('--action', choices=['convert', 'analyze', 'validate', 'visualize', 'augment'], 
                       required=True, help='Action to perform')
    parser.add_argument('--input_dir', type=str, help='Input directory (for convert action)')
    parser.add_argument('--data_dir', type=str, help='Data directory (for other actions)')
    parser.add_argument('--slice_id', type=str, help='Slice ID (for visualize/augment actions)')
    parser.add_argument('--output_dir', type=str, help='Output directory (for convert action)')
    
    args = parser.parse_args()
    
    if args.action == 'convert':
        if not args.input_dir or not args.output_dir:
            print("Error: --input_dir and --output_dir required for convert action")
            return
        convert_brats_to_yolo_format(args.input_dir, args.output_dir)
    
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
    # Example usage
    print("BraTS Dataset Processing Utilities")
    print("\nUsage examples:")
    print("1. Convert BraTS NIfTI to YOLO format:")
    print("   python data_utils.py --action convert --input_dir /path/to/BraTS2020 --output_dir /path/to/output")
    print("\n2. Analyze dataset statistics:")
    print("   python data_utils.py --action analyze --data_dir /path/to/processed_data")
    print("\n3. Validate dataset:")
    print("   python data_utils.py --action validate --data_dir /path/to/processed_data")
    print("\n4. Visualize a slice:")
    print("   python data_utils.py --action visualize --data_dir /path/to/processed_data --slice_id BraTS20_Training_002_slice_029")
    print("\n5. Visualize augmentations:")
    print("   python data_utils.py --action augment --data_dir /path/to/processed_data --slice_id BraTS20_Training_002_slice_029")
    
    # If no arguments provided, show help
    import sys
    if len(sys.argv) == 1:
        main()