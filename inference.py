"""
Inference script for Multimodal PK-YOLO brain tumor detection
"""

import torch
import cv2
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict
import argparse

from multimodal_pk_yolo import MultimodalPKYOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainTumorDetector:
    """Brain tumor detection inference class"""
    
    def __init__(self, model_path: str, device: str = 'cuda', confidence_threshold: float = 0.25, nms_threshold: float = 0.45):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        self.model = self.load_model(model_path)
        self.model.eval()
        
        logger.info(f"Model loaded on {device}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        logger.info(f"NMS threshold: {nms_threshold}")
        
    def load_model(self, model_path: str):
        """Load trained model"""
        model = MultimodalPKYOLO(num_classes=1, input_channels=4)
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model file {model_path} not found. Using random weights.")
            
        return model.to(self.device)
    
    def preprocess_multimodal_image(self, image_paths: Dict[str, str], img_size: int = 640) -> torch.Tensor:
        """
        Preprocess multimodal MRI images
        
        Args:
            image_paths: Dict with keys ['t1', 't1ce', 't2', 'flair'] and paths as values
            img_size: Target image size
        """
        modalities = ['t1', 't1ce', 't2', 'flair']
        images = []
        
        for modality in modalities:
            if modality in image_paths and Path(image_paths[modality]).exists():
                img = cv2.imread(image_paths[modality], cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))
            else:
                logger.warning(f"Missing {modality} image, using zeros")
                img = np.zeros((img_size, img_size), dtype=np.uint8)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            images.append(img)
        
        # Stack to create 4-channel image and add batch dimension
        multimodal_img = np.stack(images, axis=0)  # (4, H, W)
        tensor = torch.from_numpy(multimodal_img).unsqueeze(0).to(self.device)  # (1, 4, H, W)
        
        return tensor
    
    def post_process_predictions(self, predictions: List) -> List[Dict]:
        """
        Post-process model predictions to extract bounding boxes
        
        Args:
            predictions: List of (cls_score, bbox_pred, objectness) tuples for each scale
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # Anchors for each scale
        anchors = [
            [[10, 13], [16, 30], [33, 23]],      # P3/8
            [[30, 61], [62, 45], [59, 119]],     # P4/16  
            [[116, 90], [156, 198], [373, 326]]  # P5/32
        ]
        strides = [8, 16, 32]
        
        for scale_idx, (cls_score, bbox_pred, objectness) in enumerate(predictions):
            if scale_idx >= len(anchors):
                continue
                
            batch_size, _, h, w = cls_score.shape
            stride = strides[scale_idx]
            scale_anchors = torch.tensor(anchors[scale_idx], device=cls_score.device)
            
            # Reshape predictions
            num_anchors = 3
            cls_score = cls_score.view(batch_size, num_anchors, 1, h, w).permute(0, 1, 3, 4, 2).contiguous()
            bbox_pred = bbox_pred.view(batch_size, num_anchors, 4, h, w).permute(0, 1, 3, 4, 2).contiguous()
            objectness = objectness.view(batch_size, num_anchors, h, w)
            
            # Apply sigmoid to get probabilities
            cls_prob = torch.sigmoid(cls_score)
            obj_prob = torch.sigmoid(objectness)
            
            # Process each anchor and spatial location
            for b in range(batch_size):
                for a in range(num_anchors):
                    for i in range(h):
                        for j in range(w):
                            obj_conf = obj_prob[b, a, i, j].item()
                            
                            if obj_conf > self.confidence_threshold:
                                # Get class confidence
                                cls_conf = cls_prob[b, a, i, j, 0].item()
                                total_conf = obj_conf * cls_conf
                                
                                if total_conf > self.confidence_threshold:
                                    # Convert bbox predictions to actual coordinates
                                    bbox = bbox_pred[b, a, i, j]
                                    
                                    # Apply sigmoid and scale
                                    xy = torch.sigmoid(bbox[0:2]) * 2.0 - 0.5
                                    wh = (torch.sigmoid(bbox[2:4]) * 2) ** 2 * scale_anchors[a]
                                    
                                    # Convert to image coordinates
                                    x_center = (xy[0] + j) * stride / 640
                                    y_center = (xy[1] + i) * stride / 640
                                    width = wh[0] / 640
                                    height = wh[1] / 640
                                    
                                    detections.append({
                                        'bbox': [x_center.item(), y_center.item(), width.item(), height.item()],
                                        'confidence': total_conf,
                                        'class_id': 0,  # Tumor class
                                        'scale': scale_idx
                                    })
        
        return detections
    
    def apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression to detections"""
        if not detections:
            return []
        
        # Convert to tensors for NMS
        boxes = []
        scores = []
        
        for det in detections:
            x_center, y_center, width, height = det['bbox']
            # Convert to corner format for NMS
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            boxes.append([x1, y1, x2, y2])
            scores.append(det['confidence'])
        
        boxes = torch.tensor(boxes, device=self.device)
        scores = torch.tensor(scores, device=self.device)
        
        # Apply NMS
        keep_indices = torch.ops.torchvision.nms(boxes, scores, self.nms_threshold)
        
        # Filter detections
        filtered_detections = [detections[i] for i in keep_indices.cpu().numpy()]
        
        return filtered_detections
    
    def predict(self, image_paths: Dict[str, str]) -> List[Dict]:
        """
        Run inference on multimodal images
        
        Args:
            image_paths: Dict with modality names as keys and file paths as values
            
        Returns:
            List of detection dictionaries
        """
        with torch.no_grad():
            # Preprocess
            input_tensor = self.preprocess_multimodal_image(image_paths)
            
            # Forward pass
            predictions = self.model(input_tensor)
            
            # Post-process
            detections = self.post_process_predictions(predictions)
            
            # Apply NMS
            final_detections = self.apply_nms(detections)
            
            return final_detections

def visualize_predictions(image_path: str, detections: List[Dict], save_path: str = None):
    """
    Visualize predictions on an image
    
    Args:
        image_path: Path to reference image (e.g., T1ce)
        detections: List of detection dictionaries
        save_path: Path to save visualization
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Could not load image: {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img, cmap='gray')
    
    # Draw detections
    colors = ['red', 'blue', 'green', 'yellow', 'purple']
    
    for i, det in enumerate(detections):
        x_center, y_center, width, height = det['bbox']
        confidence = det['confidence']
        
        # Convert normalized coordinates to pixel coordinates
        x_center *= w
        y_center *= h
        width *= w
        height *= h
        
        # Convert to corner format
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        
        # Create rectangle
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor=colors[i % len(colors)],
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(x1, y1 - 5, f'Tumor: {confidence:.2f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i % len(colors)], alpha=0.7),
                fontsize=10, color='white')
    
    ax.set_title(f'Brain Tumor Detection Results ({len(detections)} detections)')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Visualization saved to {save_path}")
    
    plt.show()

def run_inference():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Brain Tumor Detection Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--t1', type=str, required=True, help='Path to T1 image')
    parser.add_argument('--t1ce', type=str, required=True, help='Path to T1ce image')
    parser.add_argument('--t2', type=str, required=True, help='Path to T2 image')
    parser.add_argument('--flair', type=str, required=True, help='Path to FLAIR image')
    parser.add_argument('--output', type=str, help='Path to save visualization')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.45, help='NMS threshold')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = BrainTumorDetector(
        model_path=args.model_path,
        device=args.device,
        confidence_threshold=args.confidence,
        nms_threshold=args.nms_threshold
    )
    
    # Prepare image paths
    image_paths = {
        't1': args.t1,
        't1ce': args.t1ce,
        't2': args.t2,
        'flair': args.flair
    }
    
    # Validate image paths
    for modality, path in image_paths.items():
        if not Path(path).exists():
            logger.error(f"{modality} image not found: {path}")
            return
    
    # Run inference
    logger.info("Running inference...")
    detections = detector.predict(image_paths)
    
    # Print results
    logger.info(f"Found {len(detections)} tumor detections:")
    for i, det in enumerate(detections):
        logger.info(f"Detection {i+1}: Confidence={det['confidence']:.3f}, "
                   f"BBox=({det['bbox'][0]:.3f}, {det['bbox'][1]:.3f}, {det['bbox'][2]:.3f}, {det['bbox'][3]:.3f})")
    
    # Visualize results
    output_path = args.output or 'detection_results.png'
    visualize_predictions(args.t1ce, detections, save_path=output_path)
    
    logger.info("Inference completed")

if __name__ == "__main__":
    run_inference()