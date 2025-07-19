"""
Specific test for class 0 handling with negative samples
This ensures empty labels don't get confused with class 0 (tumor)
"""

import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_class_zero_handling():
    """Test that negative samples don't get confused with class 0"""
    
    logger.info("Testing class 0 handling for negative samples...")
    
    try:
        from brats_dataset import collate_fn
        
        # Simulate dataset samples
        # Sample 1: Negative (no tumor)
        negative_sample = {
            'image': torch.randn(4, 640, 640),
            'bboxes': np.zeros((0, 4), dtype=np.float32),  # Empty
            'class_labels': np.array([], dtype=np.int64),  # Empty, not zeros!
            'slice_id': 'negative_slice'
        }
        
        # Sample 2: Positive (has tumor with class 0)
        positive_sample = {
            'image': torch.randn(4, 640, 640),
            'bboxes': np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
            'class_labels': np.array([0], dtype=np.int64),  # Class 0 = tumor
            'slice_id': 'positive_slice'
        }
        
        # Test collate function
        batch = [negative_sample, positive_sample]
        collated = collate_fn(batch)
        
        logger.info("Batch composition:")
        logger.info(f"  Images shape: {collated['images'].shape}")
        logger.info(f"  Bboxes shape: {collated['bboxes'].shape}")
        logger.info(f"  Labels shape: {collated['labels'].shape}")
        
        # Check negative sample (should have no valid objects)
        neg_bboxes = collated['bboxes'][0]
        neg_labels = collated['labels'][0]
        neg_has_objects = (neg_bboxes.sum() > 0).item()
        neg_valid_labels = (neg_labels >= 0).sum().item()
        
        logger.info(f"Negative sample:")
        logger.info(f"  Has objects: {neg_has_objects}")
        logger.info(f"  Valid labels: {neg_valid_labels}")
        logger.info(f"  Labels: {neg_labels}")
        
        # Check positive sample (should have class 0 tumor)
        pos_bboxes = collated['bboxes'][1]
        pos_labels = collated['labels'][1]
        pos_has_objects = (pos_bboxes.sum() > 0).item()
        pos_valid_labels = (pos_labels >= 0).sum().item()
        pos_tumor_class = pos_labels[0].item() if pos_valid_labels > 0 else None
        
        logger.info(f"Positive sample:")
        logger.info(f"  Has objects: {pos_has_objects}")
        logger.info(f"  Valid labels: {pos_valid_labels}")
        logger.info(f"  Tumor class: {pos_tumor_class}")
        logger.info(f"  Labels: {pos_labels}")
        
        # Verify correct handling
        success = True
        
        if neg_has_objects:
            logger.error("❌ Negative sample incorrectly shows objects!")
            success = False
        
        if neg_valid_labels > 0:
            logger.error("❌ Negative sample has valid labels (should be all -1)!")
            success = False
            
        if not pos_has_objects:
            logger.error("❌ Positive sample missing objects!")
            success = False
            
        if pos_tumor_class != 0:
            logger.error(f"❌ Positive sample wrong class! Expected 0, got {pos_tumor_class}")
            success = False
        
        if success:
            logger.info("✅ Class 0 handling test PASSED!")
            logger.info("  - Negative samples have no objects (correct)")
            logger.info("  - Positive samples have class 0 tumors (correct)")
            logger.info("  - No confusion between empty and class 0")
        else:
            logger.error("❌ Class 0 handling test FAILED!")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_with_class_zero():
    """Test loss function with class 0 setup"""
    
    logger.info("Testing loss function with class 0...")
    
    try:
        from multimodal_pk_yolo import MultimodalPKYOLO
        from complete_yolo_loss import YOLOLoss
        
        model = MultimodalPKYOLO(num_classes=1, input_channels=4)
        loss_fn = YOLOLoss(model, num_classes=1)
        
        batch_size = 2
        images = torch.randn(batch_size, 4, 640, 640)
        predictions = model(images)
        
        # Test negative samples (background - no tumors)
        negative_targets = {
            'images': images,
            'bboxes': torch.zeros(batch_size, 1, 4),  # No valid bboxes
            'labels': torch.full((batch_size, 1), -1, dtype=torch.long)  # -1 padding
        }
        
        loss_neg, loss_components_neg = loss_fn(predictions, negative_targets)
        
        logger.info(f"Negative samples (no tumors):")
        logger.info(f"  Total loss: {loss_neg:.6f}")
        logger.info(f"  Box: {loss_components_neg[0]:.6f} (should be 0)")
        logger.info(f"  Obj: {loss_components_neg[1]:.6f} (should be > 0)")
        logger.info(f"  Cls: {loss_components_neg[2]:.6f} (should be 0)")
        
        # Test positive samples (tumors with class 0)
        positive_targets = {
            'images': images,
            'bboxes': torch.tensor([
                [[0.5, 0.5, 0.2, 0.2]],  # Tumor at center
                [[0.3, 0.7, 0.15, 0.15]]  # Another tumor
            ], dtype=torch.float32),
            'labels': torch.tensor([
                [0],  # Class 0 (tumor)
                [0]   # Class 0 (tumor)
            ], dtype=torch.long)
        }
        
        loss_pos, loss_components_pos = loss_fn(predictions, positive_targets)
        
        logger.info(f"Positive samples (class 0 tumors):")
        logger.info(f"  Total loss: {loss_pos:.6f}")
        logger.info(f"  Box: {loss_components_pos[0]:.6f} (should be > 0)")
        logger.info(f"  Obj: {loss_components_pos[1]:.6f} (should be > 0)")
        logger.info(f"  Cls: {loss_components_pos[2]:.6f} (should be > 0)")
        
        # Verify results
        success = True
        
        # For negative samples
        if loss_components_neg[1].item() <= 0:
            logger.error("❌ Objectness loss should be > 0 for negative samples!")
            success = False
            
        if loss_components_neg[0].item() != 0:
            logger.warning("⚠️  Box loss should be 0 for negative samples")
            
        # For positive samples  
        if loss_pos.item() <= 0:
            logger.error("❌ Total loss should be > 0 for positive samples!")
            success = False
            
        if success:
            logger.info("✅ Loss function test PASSED!")
        else:
            logger.error("❌ Loss function test FAILED!")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all class 0 specific tests"""
    
    logger.info("🔍 TESTING CLASS 0 HANDLING")
    logger.info("=" * 50)
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Collate function
    logger.info("\n1. Testing collate function with class 0...")
    if test_class_zero_handling():
        tests_passed += 1
    
    # Test 2: Loss function
    logger.info("\n2. Testing loss function with class 0...")
    if test_loss_with_class_zero():
        tests_passed += 1
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        logger.info("🎉 All class 0 tests PASSED!")
        logger.info("Your setup correctly handles:")
        logger.info("  ✅ Empty labels as negative samples")
        logger.info("  ✅ Class 0 as tumor class")
        logger.info("  ✅ No confusion between empty and class 0")
    else:
        logger.error("❌ Some tests FAILED!")
        logger.error("Please check the implementation before training.")

if __name__ == "__main__":
    main()