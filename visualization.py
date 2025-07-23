"""
Visualization utilities for training monitoring
Saves PNG images with ground truth and predicted bounding boxes
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

class TrainingVisualizer:
    """Visualizer for training progress with GT and prediction overlays"""
    
    def __init__(self, output_dir: str, save_interval: int = 100):
        self.output_dir = Path(output_dir)
        self.save_interval = save_interval
        self.batch_count = 0
        
        # Create visualization directory
        self.vis_dir = self.output_dir / 'training_visualizations'
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Training visualizer initialized. Saving every {save_interval} batches to {self.vis_dir}")
    
    def should_save(self, batch_idx: int) -> bool:
        """Check if we should save visualization for this batch"""
        return batch_idx % self.save_interval == 0
    
    def denormalize_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Denormalize image tensor back to [0, 255] range
        Assumes input is normalized with ImageNet stats
        """
        # ImageNet normalization stats for 4 channels (repeated for FLAIR)
        mean = torch.tensor([0.485, 0.456, 0.406, 0.485]).view(1, 4, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225, 0.229]).view(1, 4, 1, 1)
        
        if image_tensor.is_cuda:
            mean = mean.cuda()
            std = std.cuda()
        
        # Denormalize
        image = image_tensor * std + mean
        image = torch.clamp(image, 0, 1)
        
        # Convert to numpy and scale to [0, 255]
        image_np = image.cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        return image_np
    
    def create_composite_image(self, multimodal_image: torch.Tensor, modality_idx: int = 1) -> np.ndarray:
        """
        Create a composite visualization from multimodal image
        
        Args:
            multimodal_image: Tensor of shape (C, H, W) with 4 channels
            modality_idx: Which modality to use as base (0=T1, 1=T1ce, 2=T2, 3=FLAIR)
        """
        if multimodal_image.dim() == 4:
            multimodal_image = multimodal_image[0]  # Remove batch dimension
        
        # Denormalize
        image_np = self.denormalize_image(multimodal_image.unsqueeze(0))[0]
        
        # Use specified modality as base
        base_image = image_np[modality_idx]  # Shape: (H, W)
        
        # Create RGB image (grayscale repeated)
        rgb_image = np.stack([base_image, base_image, base_image], axis=-1)
        
        return rgb_image
    
    def decode_predictions(self, predictions: List[Tuple], img_size: int = 640, 
                         confidence_threshold: float = 0.1) -> List[Dict]:
        """
        Decode model predictions to bounding boxes
        
        Args:
            predictions: List of (cls_score, bbox_pred, objectness) for each scale
            img_size: Image size for coordinate conversion
            confidence_threshold: Minimum confidence for detection
        
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # Anchors for each scale (copied from model)
        anchors = torch.tensor([
            [[10, 13], [16, 30], [33, 23]],      # P3/8
            [[30, 61], [62, 45], [59, 119]],     # P4/16  
            [[116, 90], [156, 198], [373, 326]]  # P5/32
        ], dtype=torch.float32)
        
        strides = [8, 16, 32]  # Strides for each scale
        
        for scale_idx, (cls_score, bbox_pred, objectness) in enumerate(predictions):
            if scale_idx >= len(anchors):
                continue
                
            batch_size, _, h, w = cls_score.shape
            stride = strides[scale_idx]
            scale_anchors = anchors[scale_idx]
            
            if cls_score.is_cuda:
                scale_anchors = scale_anchors.cuda()
            
            # Reshape predictions
            num_classes = 1  # Brain tumor detection
            num_anchors = 3
            
            cls_score = cls_score.view(batch_size, num_anchors, num_classes, h, w).permute(0, 1, 3, 4, 2)
            bbox_pred = bbox_pred.view(batch_size, num_anchors, 4, h, w).permute(0, 1, 3, 4, 2)
            objectness = objectness.view(batch_size, num_anchors, h, w)
            
            # Apply sigmoid
            cls_prob = torch.sigmoid(cls_score)
            obj_prob = torch.sigmoid(objectness)
            
            # Process first image in batch only
            for a in range(num_anchors):
                for i in range(h):
                    for j in range(w):
                        obj_conf = obj_prob[0, a, i, j].item()
                        
                        if obj_conf > confidence_threshold:
                            # Get class confidence (assuming single class)
                            cls_conf = cls_prob[0, a, i, j, 0].item()
                            total_conf = obj_conf * cls_conf
                            
                            if total_conf > confidence_threshold:
                                # Decode bounding box
                                bbox = bbox_pred[0, a, i, j]
                                
                                # Apply sigmoid and scale
                                xy = torch.sigmoid(bbox[0:2]) * 2.0 - 0.5
                                wh = (torch.sigmoid(bbox[2:4]) * 2) ** 2 * scale_anchors[a]
                                
                                # Convert to image coordinates
                                x_center = (xy[0] + j) * stride
                                y_center = (xy[1] + i) * stride
                                width = wh[0]
                                height = wh[1]
                                
                                # Normalize to [0, 1]
                                x_center_norm = x_center / img_size
                                y_center_norm = y_center / img_size
                                width_norm = width / img_size
                                height_norm = height / img_size
                                
                                detections.append({
                                    'bbox': [x_center_norm.item(), y_center_norm.item(), 
                                           width_norm.item(), height_norm.item()],
                                    'confidence': total_conf,
                                    'class_id': 0,
                                    'scale': scale_idx
                                })
        
        return detections
    
    def draw_bounding_boxes(self, image: np.ndarray, gt_boxes: List[List[float]], 
                          pred_boxes: List[Dict], img_size: int = 640) -> np.ndarray:
        """
        Draw ground truth and predicted bounding boxes on image
        
        Args:
            image: RGB image array
            gt_boxes: List of ground truth boxes in [x_center, y_center, width, height] format
            pred_boxes: List of prediction dictionaries
            img_size: Image size for coordinate conversion
        """
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(image, cmap='gray')
        
        h, w = image.shape[:2]
        
        # Draw ground truth boxes (GREEN)
        for gt_box in gt_boxes:
            if len(gt_box) >= 4 and sum(gt_box) > 0:  # Valid box
                x_center, y_center, width, height = gt_box[:4]
                
                # Convert to pixel coordinates
                x_center_px = x_center * w
                y_center_px = y_center * h
                width_px = width * w
                height_px = height * h
                
                # Convert to corner format
                x1 = x_center_px - width_px / 2
                y1 = y_center_px - height_px / 2
                
                # Create rectangle
                rect = Rectangle(
                    (x1, y1), width_px, height_px,
                    linewidth=3,
                    edgecolor='lime',
                    facecolor='none',
                    alpha=0.8
                )
                ax.add_patch(rect)
                
                # Add label
                ax.text(x1, y1 - 5, 'GT Tumor', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lime', alpha=0.8),
                       fontsize=10, color='black', weight='bold')
        
        # Draw predicted boxes (RED)
        for pred in pred_boxes:
            x_center, y_center, width, height = pred['bbox']
            confidence = pred['confidence']
            
            # Convert to pixel coordinates
            x_center_px = x_center * w
            y_center_px = y_center * h
            width_px = width * w
            height_px = height * h
            
            # Convert to corner format
            x1 = x_center_px - width_px / 2
            y1 = y_center_px - height_px / 2
            
            # Create rectangle
            rect = Rectangle(
                (x1, y1), width_px, height_px,
                linewidth=3,
                edgecolor='red',
                facecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add label with confidence
            ax.text(x1, y1 - 25, f'Pred: {confidence:.2f}', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.8),
                   fontsize=10, color='white', weight='bold')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='lime', lw=3, label='Ground Truth'),
            Line2D([0], [0], color='red', lw=3, label='Predictions')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        ax.set_title(f'Training Visualization - GT: {len(gt_boxes)}, Pred: {len(pred_boxes)}', 
                    fontsize=14, weight='bold')
        ax.axis('off')
        
        # Convert plot to numpy array
        fig.canvas.draw()
        plot_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_array = plot_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return plot_array
    
    def save_training_visualization(self, batch_idx: int, epoch: int, 
                                  images: torch.Tensor, targets: Dict,
                                  predictions: List[Tuple], slice_ids: List[str]):
        """
        Save visualization of training batch with GT and predictions
        
        Args:
            batch_idx: Current batch index
            epoch: Current epoch
            images: Batch of input images
            targets: Dictionary with 'bboxes' and 'labels'
            predictions: Model predictions
            slice_ids: List of slice identifiers
        """
        try:
            # Process first image in batch only
            if len(images) == 0:
                return
            
            image = images[0]  # Shape: (4, H, W)
            gt_bboxes = targets['bboxes'][0]  # Shape: (N, 4)
            gt_labels = targets['labels'][0]  # Shape: (N,)
            slice_id = slice_ids[0] if slice_ids else f"unknown_{batch_idx}"
            
            # Filter valid ground truth boxes (labels >= 0)
            valid_mask = gt_labels >= 0
            valid_gt_boxes = gt_bboxes[valid_mask].cpu().numpy().tolist()
            
            # Create composite image (using T1ce modality)
            composite_image = self.create_composite_image(image, modality_idx=1)
            
            # Decode predictions
            pred_boxes = self.decode_predictions(predictions, img_size=image.shape[-1])
            
            # Draw bounding boxes
            vis_image = self.draw_bounding_boxes(
                composite_image, valid_gt_boxes, pred_boxes, img_size=image.shape[-1]
            )
            
            # Save image
            filename = f"epoch_{epoch:03d}_batch_{batch_idx:04d}_{slice_id}.png"
            save_path = self.vis_dir / filename
            
            plt.imsave(save_path, vis_image)
            
            # Log summary
            logger.info(f"Saved visualization: {filename} | GT boxes: {len(valid_gt_boxes)} | "
                       f"Predictions: {len(pred_boxes)} | Slice: {slice_id}")
            
        except Exception as e:
            logger.error(f"Error saving visualization for batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    def update_batch_count(self, batch_idx: int, epoch: int, images: torch.Tensor, 
                          targets: Dict, predictions: List[Tuple], slice_ids: List[str]):
        """
        Update batch count and save visualization if needed
        
        Args:
            batch_idx: Current batch index within epoch
            epoch: Current epoch
            images: Batch of input images
            targets: Dictionary with 'bboxes' and 'labels'  
            predictions: Model predictions
            slice_ids: List of slice identifiers
        """
        self.batch_count += 1
        
        # Save every save_interval batches
        if self.batch_count % self.save_interval == 0:
            logger.info(f"Creating training visualization at epoch {epoch}, batch {batch_idx}")
            self.save_training_visualization(
                batch_idx, epoch, images, targets, predictions, slice_ids
            )

def apply_nms_to_predictions(detections: List[Dict], nms_threshold: float = 0.45) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to filter overlapping detections
    
    Args:
        detections: List of detection dictionaries
        nms_threshold: IoU threshold for NMS
    
    Returns:
        Filtered list of detections
    """
    if not detections:
        return []
    
    # Convert to tensor format for NMS
    boxes = []
    scores = []
    
    for det in detections:
        x_center, y_center, width, height = det['bbox']
        # Convert to corner format
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        boxes.append([x1, y1, x2, y2])
        scores.append(det['confidence'])
    
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    
    # Apply NMS
    try:
        keep_indices = torch.ops.torchvision.nms(boxes, scores, nms_threshold)
        filtered_detections = [detections[i] for i in keep_indices.numpy()]
    except:
        # Fallback: simple confidence-based filtering
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        filtered_detections = sorted_detections[:10]  # Keep top 10
    
    return filtered_detections

# Test visualization functionality
def test_visualizer():
    """Test the training visualizer"""
    import torch
    
    logger.info("Testing training visualizer...")
    
    # Create dummy data
    batch_size = 2
    img_size = 640
    
    # Dummy images (4 channels)
    images = torch.randn(batch_size, 4, img_size, img_size)
    
    # Dummy targets
    targets = {
        'bboxes': torch.tensor([
            [[0.5, 0.5, 0.2, 0.3], [0.0, 0.0, 0.0, 0.0]],  # First image has 1 box
            [[0.3, 0.7, 0.15, 0.25], [0.8, 0.2, 0.1, 0.1]]  # Second image has 2 boxes
        ]),
        'labels': torch.tensor([
            [0, -1],  # First image: 1 tumor, 1 padding
            [0, 0]    # Second image: 2 tumors
        ])
    }
    
    # Dummy predictions (simplified)
    predictions = [
        (torch.randn(batch_size, 3, img_size//8, img_size//8),   # cls_score
         torch.randn(batch_size, 12, img_size//8, img_size//8),  # bbox_pred (3 anchors * 4 coords)
         torch.randn(batch_size, 3, img_size//8, img_size//8)),  # objectness
    ]
    
    slice_ids = ['test_slice_001', 'test_slice_002']
    
    # Test visualizer
    visualizer = TrainingVisualizer('test_output', save_interval=1)
    
    # Test saving
    visualizer.update_batch_count(
        batch_idx=0, epoch=1, images=images, 
        targets=targets, predictions=predictions, slice_ids=slice_ids
    )
    
    logger.info("✅ Visualizer test completed!")

if __name__ == "__main__":
    test_visualizer()