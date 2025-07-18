"""
Quick fix script to resolve the zero box/class loss issue
Run this to patch your training immediately
"""

import torch
import logging

logger = logging.getLogger(__name__)

def check_label_format(data_dir):
    """Check the format of your label files"""
    from pathlib import Path
    
    print("🔍 Checking label format...")
    
    label_dir = Path(data_dir) / 'train' / 'labels'
    label_files = list(label_dir.glob('*.txt'))
    
    if not label_files:
        print("❌ No label files found!")
        return False
    
    # Check first few label files
    for i, label_file in enumerate(label_files[:3]):
        print(f"\n📄 Checking {label_file.name}:")
        
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                print("   Empty file")
                continue
            
            for line_num, line in enumerate(lines[:3]):  # Check first 3 lines
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    print(f"   Line {line_num + 1}: class={class_id}, bbox=({x:.3f}, {y:.3f}, {w:.3f}, {h:.3f})")
                    
                    # Check if coordinates are valid
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        print(f"   ⚠️  Invalid coordinates (should be 0-1): {line.strip()}")
                else:
                    print(f"   ⚠️  Invalid format: {line.strip()}")
        
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    
    return True

def debug_target_assignment(data_dir):
    """Debug target assignment during training"""
    print("🔍 Testing target assignment...")
    
    try:
        # Import your classes
        from brats_dataset import BraTSDataset, collate_fn
        from multimodal_pk_yolo import MultimodalPKYOLO
        from complete_yolo_loss import YOLOLoss
        from torch.utils.data import DataLoader
        
        # Create dataset
        dataset = BraTSDataset(data_dir, split='train', img_size=640, augment=False)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
        
        # Create model and loss
        model = MultimodalPKYOLO(num_classes=1, input_channels=4)
        criterion = YOLOLoss(model, num_classes=1, autobalance=True)
        
        # Get one batch
        batch = next(iter(dataloader))
        
        print(f"Batch info:")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  Bboxes shape: {batch['bboxes'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        
        # Check labels in batch
        for i in range(batch['bboxes'].shape[0]):
            bboxes = batch['bboxes'][i]
            labels = batch['labels'][i]
            
            valid_mask = (bboxes.sum(dim=1) > 0) & (labels >= 0)
            valid_bboxes = bboxes[valid_mask]
            valid_labels = labels[valid_mask]
            
            print(f"\nImage {i}:")
            print(f"  Valid boxes: {len(valid_bboxes)}")
            print(f"  Valid labels: {valid_labels.tolist()}")
            if len(valid_bboxes) > 0:
                print(f"  Box sizes (w,h): {valid_bboxes[:, 2:4].tolist()}")
        
        # Test forward pass
        images = batch['images'].float()
        targets = {
            'bboxes': batch['bboxes'].float(),
            'labels': batch['labels'].long(),
            'images': images
        }
        
        # Enable debug logging
        logging.getLogger().setLevel(logging.DEBUG)
        
        with torch.no_grad():
            predictions = model(images)
            loss, loss_components = criterion(predictions, targets)
        
        print(f"\nLoss Results:")
        print(f"  Total loss: {loss.item():.6f}")
        print(f"  Box loss: {loss_components[0]:.6f}")
        print(f"  Obj loss: {loss_components[1]:.6f}")
        print(f"  Cls loss: {loss_components[2]:.6f}")
        
        if loss_components[0] == 0 and loss_components[2] == 0:
            print("\n❌ Still getting zero box/class losses!")
            print("This suggests target assignment is failing.")
            return False
        else:
            print("\n✅ Loss calculation looks good!")
            return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_label_indexing(data_dir):
    """Fix label indexing if needed"""
    print("🔧 Checking and fixing label indexing...")
    
    from pathlib import Path
    
    label_dirs = [
        Path(data_dir) / 'train' / 'labels',
        Path(data_dir) / 'val' / 'labels',
        Path(data_dir) / 'test' / 'labels'
    ]
    
    for label_dir in label_dirs:
        if not label_dir.exists():
            continue
        
        print(f"Processing {label_dir}...")
        label_files = list(label_dir.glob('*.txt'))
        
        fixed_count = 0
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                modified = False
                new_lines = []
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # If class is 1, convert to 0 for single-class detection
                        if class_id == 1:
                            parts[0] = '0'
                            modified = True
                        elif class_id > 1:
                            print(f"⚠️  Found class {class_id} in {label_file.name} - this might indicate multi-class labels")
                    
                    new_lines.append(' '.join(parts) + '\n')
                
                if modified:
                    # Backup original file
                    backup_file = label_file.with_suffix('.txt.backup')
                    if not backup_file.exists():
                        label_file.rename(backup_file)
                    
                    # Write fixed file
                    with open(label_file, 'w') as f:
                        f.writelines(new_lines)
                    
                    fixed_count += 1
            
            except Exception as e:
                print(f"Error processing {label_file}: {e}")
        
        print(f"  Fixed {fixed_count} files in {label_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to your dataset')
    parser.add_argument('--action', type=str, choices=['check', 'debug', 'fix', 'all'], 
                       default='all', help='Action to perform')
    
    args = parser.parse_args()
    
    print("🔧 FIXING ZERO LOSS ISSUE")
    print("=" * 40)
    
    if args.action in ['check', 'all']:
        check_label_format(args.data_dir)
    
    if args.action in ['fix', 'all']:
        fix_label_indexing(args.data_dir)
    
    if args.action in ['debug', 'all']:
        debug_target_assignment(args.data_dir)
    
    print("\n💡 SOLUTIONS:")
    print("1. Replace 'complete_yolo_loss.py' with 'fixed_single_class_loss.py'")
    print("2. In your training script, change the import:")
    print("   from fixed_single_class_loss import YOLOLoss")
    print("3. Restart training with smaller anchors for brain tumors")

if __name__ == "__main__":
    main()