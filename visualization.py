"""
Simple debug visualization - just save images
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class DebugVisualizer:
    def __init__(self, output_dir: str, save_interval: int = 100):
        self.vis_dir = Path(output_dir) / 'debug_visualizations'
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.count = 0
        print(f"🎨 Debug visualizer ready: {self.vis_dir}")
    
    def save_batch_debug(self, batch_idx: int, epoch: int, images: torch.Tensor, targets: dict):
        """Save simple debug visualization"""
        if batch_idx % self.save_interval != 0:
            return
            
        try:
            # Take first image and first channel (T1ce)
            img = images[0, 1].detach().cpu().numpy()  # Channel 1 = T1ce
            
            # Normalize to 0-255
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)
            
            # Get ground truth boxes
            gt_boxes = targets['bboxes'][0].detach().cpu().numpy()
            gt_labels = targets['labels'][0].detach().cpu().numpy()
            
            # Create plot
            plt.figure(figsize=(8, 8))
            plt.imshow(img, cmap='gray')
            
            # Draw GT boxes
            h, w = img.shape
            for i, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
                if label >= 0:  # Valid annotation
                    x_center, y_center, width, height = box
                    x1 = (x_center - width/2) * w
                    y1 = (y_center - height/2) * h
                    w_box = width * w
                    h_box = height * h
                    
                    rect = plt.Rectangle((x1, y1), w_box, h_box, 
                                       fill=False, color='lime', linewidth=2)
                    plt.gca().add_patch(rect)
            
            plt.title(f'Epoch {epoch}, Batch {batch_idx}')
            plt.axis('off')
            
            # Save
            filename = f'debug_epoch_{epoch:03d}_batch_{batch_idx:04d}.png'
            save_path = self.vis_dir / filename
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close()
            
            print(f"✅ Saved: {filename}")
            
        except Exception as e:
            print(f"❌ Error saving visualization: {e}")
            import traceback
            traceback.print_exc()

# Aggiungi questo al training script:
def add_to_training_loop():
    """
    Aggiungi questo codice al tuo training loop:
    
    # Nella classe Trainer.__init__():
    self.debug_vis = DebugVisualizer(str(self.output_dir), save_interval=100)
    
    # Nel train_epoch(), dopo il forward pass:
    self.debug_vis.save_batch_debug(batch_idx, self.current_epoch, images, targets)
    """
    pass