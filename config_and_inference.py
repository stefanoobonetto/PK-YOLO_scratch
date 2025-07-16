"""
Configuration file and inference utilities for Multimodal PK-YOLO
"""

import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict

# config.yaml content
CONFIG_YAML = """
# Multimodal PK-YOLO Configuration

# Model Configuration
model:
  num_classes: 1
  input_channels: 4  # T1, T1ce, T2, FLAIR
  img_size: 640
  confidence_threshold: 0.25
  nms_threshold: 0.45

# Training Configuration
training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  warmup_epochs: 5
  
# Data Configuration
data:
  train_dir: "train"
  val_dir: "val" 
  test_dir: "test"
  num_workers: 4
  pin_memory: true

# Augmentation Configuration
augmentation:
  enabled: true
  horizontal_flip: 0.5
  brightness_contrast: 0.3
  gamma: 0.3
  noise: 0.3
  random_crop_scale: [0.8, 1.0]

# Loss Configuration
loss:
  obj_weight: 1.0
  cls_weight: 1.0
  bbox_weight: 5.0
  ignore_threshold: 0.5

# Optimizer Configuration
optimizer:
  type: "AdamW"
  lr_scheduler: "CosineAnnealingLR"
  
# Validation Configuration
validation:
  eval_interval: 10
  save_best: true
  
# Logging Configuration
logging:
  log_dir: "logs"
  save_interval: 50
  tensorboard: true
"""

class Config:
    """Configuration class for easy access to config parameters"""
    
    def __init__(self, config_path: str = None):
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = yaml.safe_load(CONFIG_YAML)
    
    def __getattr__(self, name):
        return self.config.get(name, {})
    
    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

def save_config(config_path: str = "config.yaml"):
    """Save default configuration to file"""
    with open(config_path, 'w') as f:
        f.write(CONFIG_YAML)
    print(f"Configuration saved to {config_path}")

class ModelInference:
    """Inference class for trained Multimodal PK-YOLO model"""
    
    def __init__(self, model_path: str, config: Config, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()
        
    def load_model(self, model_path: str):
        """Load trained model"""
        from multimodal_pk_yolo import MultimodalPKYOLO  # Import your main model class
        
        model = MultimodalPKYOLO(
            num_classes=self.config.get('model.num_classes', 1),
            input_channels=self.config.get('model.input_channels', 4)
        )
        
        if Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using random weights.")
            
        return model.to(self.device)
    
    def preprocess_multimodal_image(self, image_paths: Dict[str, str]) -> torch.Tensor:
        """
        Preprocess multimodal MRI images
        
        Args:
            image_paths: Dict with keys ['t1', 't1ce', 't2', 'flair'] and paths as values
        """
        modalities = ['t1', 't1ce', 't2', 'flair']
        images = []
        img_size = self.config.get('model.img_size', 640)
        
        for modality in modalities:
            if modality in image_paths and Path(image_paths[modality]).exists():
                img = cv2.imread(image_paths[modality], cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))
            else:
                print(f"Warning: Missing {modality} image, using zeros")
                img = np.zeros((img_size, img_size), dtype=np.uint8)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            images.append(img)
        
        # Stack to create 4-channel image and add batch dimension
        multimodal_img = np.stack(images, axis=0)  # (4, H, W)
        tensor = torch.from_numpy(multimodal_img).unsqueeze(0).to(self.device)  # (1, 4, H, W)
        
        return tensor
    
    def post_process_predictions(self, predictions: List[Tuple]) -> List[Dict]:
        """
        Post-process model predictions to extract bounding boxes
        
        Args:
            predictions: List of (cls_score, bbox_pred, objectness) tuples for each scale
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        confidence_threshold = self.config.get('model.confidence_threshold', 0.25)
        
        for scale_idx, (cls_score, bbox_pred, objectness) in enumerate(predictions):
            batch_size, num_anchors_times_classes, h, w = cls_score.shape
            num_anchors = num_anchors_times_classes // self.config.get('model.num_classes', 1)
            
            # Reshape predictions
            cls_score = cls_score.view(batch_size, num_anchors, -1, h, w).permute(0, 1, 3, 4, 2).contiguous()
            bbox_pred = bbox_pred.view(batch_size, num_anchors, 4, h, w).permute(0, 1, 3, 4, 2).contiguous()
            objectness = objectness.view(batch_size, num_anchors, h, w)
            
            # Apply sigmoid to get probabilities
            cls_prob = torch.sigmoid(cls_score)
            obj_prob = torch.sigmoid(objectness)
            
            # Create grid
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=-1).float().to(self.device)
            
            # Process each image in batch (assuming batch_size=1 for inference)
            for b in range(batch_size):
                for a in range(num_anchors):
                    for i in range(h):
                        for j in range(w):
                            obj_conf = obj_prob[b, a, i, j].item()
                            
                            if obj_conf > confidence_threshold:
                                # Get class confidence
                                cls_conf = cls_prob[b, a, i, j].max().item()
                                cls_id = cls_prob[b, a, i, j].argmax().item()
                                
                                total_conf = obj_conf * cls_conf
                                
                                if total_conf > confidence_threshold:
                                    # Convert bbox predictions to actual coordinates
                                    bbox = bbox_pred[b, a, i, j]
                                    
                                    # Apply sigmoid and scale to grid
                                    x_center = (torch.sigmoid(bbox[0]) + j) / w
                                    y_center = (torch.sigmoid(bbox[1]) + i) / h
                                    width = torch.sigmoid(bbox[2])
                                    height = torch.sigmoid(bbox[3])
                                    
                                    detections.append({
                                        'bbox': [x_center.item(), y_center.item(), width.item(), height.item()],
                                        'confidence': total_conf,
                                        'class_id': cls_id,
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
        nms_threshold = self.config.get('model.nms_threshold', 0.45)
        keep_indices = torch.ops.torchvision.nms(boxes, scores, nms_threshold)
        
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
        print(f"Could not load image: {image_path}")
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
        print(f"Visualization saved to {save_path}")
    
    plt.show()

class EvaluationMetrics:
    """Evaluation metrics for object detection"""
    
    def __init__(self, iou_thresholds: List[float] = None):
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes in [x_center, y_center, width, height] format"""
        # Convert to corner format
        x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
        x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
        
        x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
        x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_predictions(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
        """
        Evaluate predictions against ground truth
        
        Args:
            predictions: List of prediction dictionaries
            ground_truths: List of ground truth dictionaries
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            'mAP': 0.0,
            'mAP50': 0.0,
            'mAP75': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        
        if not ground_truths:
            return results
        
        # Calculate metrics for each IoU threshold
        aps = []
        
        for iou_thresh in self.iou_thresholds:
            tp = 0
            fp = 0
            fn = 0
            
            # Match predictions to ground truths
            matched_gt = set()
            
            for pred in predictions:
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(ground_truths):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_thresh:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            
            fn = len(ground_truths) - len(matched_gt)
            
            # Calculate precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Simple AP calculation (could be improved with PR curve)
            ap = precision * recall
            aps.append(ap)
            
            if iou_thresh == 0.5:
                results['mAP50'] = ap
                results['precision'] = precision
                results['recall'] = recall
                results['f1_score'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            elif iou_thresh == 0.75:
                results['mAP75'] = ap
        
        results['mAP'] = np.mean(aps)
        
        return results

def run_inference_example():
    """Example of how to run inference"""
    
    # Load configuration
    config = Config()
    
    # Initialize inference
    model_path = "best_multimodal_pk_yolo.pth"
    inference = ModelInference(model_path, config)
    
    # Example image paths for one slice
    image_paths = {
        't1': 'path/to/BraTS20_Training_002_slice_029_t1.png',
        't1ce': 'path/to/BraTS20_Training_002_slice_029_t1ce.png',
        't2': 'path/to/BraTS20_Training_002_slice_029_t2.png',
        'flair': 'path/to/BraTS20_Training_002_slice_029_flair.png'
    }
    
    # Run inference
    detections = inference.predict(image_paths)
    
    # Print results
    print(f"Found {len(detections)} tumor detections:")
    for i, det in enumerate(detections):
        print(f"Detection {i+1}: Confidence={det['confidence']:.3f}, "
              f"BBox={det['bbox']}, Class={det['class_id']}")
    
    # Visualize results
    visualize_predictions(
        image_paths['t1ce'], 
        detections, 
        save_path='detection_results.png'
    )

def evaluate_model_on_test_set():
    """Example of how to evaluate model on test set"""
    
    config = Config()
    
    # Load test dataset
    from multimodal_pk_yolo import BraTSDataset
    test_dataset = BraTSDataset(
        config.get('data_dir', '/path/to/data'), 
        split='test',
        img_size=config.get('model.img_size', 640),
        augment=False
    )
    
    # Initialize inference and evaluation
    model_path = "best_multimodal_pk_yolo.pth"
    inference = ModelInference(model_path, config)
    evaluator = EvaluationMetrics()
    
    all_predictions = []
    all_ground_truths = []
    
    # Run inference on test set
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        slice_id = sample['slice_id']
        
        # Create image paths (you'll need to adapt this to your data structure)
        image_paths = {
            't1': f"path/to/test/images/{slice_id}_t1.png",
            't1ce': f"path/to/test/images/{slice_id}_t1ce.png",
            't2': f"path/to/test/images/{slice_id}_t2.png",
            'flair': f"path/to/test/images/{slice_id}_flair.png"
        }
        
        # Get predictions
        predictions = inference.predict(image_paths)
        
        # Get ground truth
        ground_truths = []
        for bbox, label in zip(sample['bboxes'], sample['class_labels']):
            if label > 0:  # Valid annotation
                ground_truths.append({
                    'bbox': bbox.tolist(),
                    'class_id': label.item()
                })
        
        all_predictions.extend(predictions)
        all_ground_truths.extend(ground_truths)
    
    # Evaluate
    metrics = evaluator.evaluate_predictions(all_predictions, all_ground_truths)
    
    print("Evaluation Results:")
    print(f"mAP: {metrics['mAP']:.3f}")
    print(f"mAP@50: {metrics['mAP50']:.3f}")
    print(f"mAP@75: {metrics['mAP75']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")

if __name__ == "__main__":
    # Save default configuration
    save_config("config.yaml")
    
    # Run inference example
    # run_inference_example()
    
    # Evaluate model
    # evaluate_model_on_test_set()
    
    print("Configuration and inference utilities ready!")
    print("To use:")
    print("1. Train model with: python multimodal_pk_yolo.py")
    print("2. Run inference with: run_inference_example()")
    print("3. Evaluate model with: evaluate_model_on_test_set()")