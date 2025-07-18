"""
Fixed YOLO Loss for Single-Class Brain Tumor Detection
Resolves the zero box/class loss issue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

def smooth_BCE(eps=0.1):
    """Label smoothing for BCE targets"""
    return 1.0 - 0.5 * eps, 0.5 * eps

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """Calculate IoU between two sets of boxes"""
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if CIoU:
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU
    return iou  # IoU

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class YOLOLoss(nn.Module):
    """
    Improved YOLO Loss that properly handles negative samples (background images)
    """
    
    def __init__(self, model, num_classes=1, anchors=None, autobalance=False):
        super().__init__()
        
        device = next(model.parameters()).device
        self.device = device
        self.num_classes = num_classes
        
        # Hyperparameters for brain tumor detection with negative samples
        self.hyp = {
            'box': 0.05,           # Box loss weight (only for positive samples)
            'cls': 0.3,            # Class loss weight  
            'obj': 1.0,            # Object loss weight (crucial for negative samples)
            'anchor_t': 4.0,       # Anchor matching threshold
            'fl_gamma': 0.0,       # Focal loss gamma
            'cls_pw': 1.0,         # Class positive weight
            'obj_pw': 1.0,         # Object positive weight
            'label_smoothing': 0.0, # Label smoothing epsilon
        }
        
        # Loss functions
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['obj_pw']], device=device))
        
        self.BCEcls = BCEcls
        self.BCEobj = BCEobj
        
        # Label smoothing
        self.cp, self.cn = self.smooth_BCE(eps=self.hyp.get('label_smoothing', 0.0))
        
        # Anchors optimized for brain tumors
        if anchors is None:
            self.anchors = torch.tensor([
                [[6, 8], [10, 12], [16, 20]],        # P3/8 - small tumors
                [[20, 25], [30, 35], [40, 50]],      # P4/16 - medium tumors  
                [[60, 70], [80, 100], [120, 150]]    # P5/32 - larger tumors
            ], dtype=torch.float32, device=device)
        else:
            self.anchors = anchors.to(device)
        
        self.nl = len(self.anchors)
        self.na = self.anchors.shape[1]
        
        # Balance weights - all scales important for negative samples
        self.balance = [4.0, 1.0, 0.25] if self.nl == 3 else [1.0] * self.nl
        self.autobalance = autobalance
        self.gr = 1.0
        
        logger.info(f"ImprovedYOLOLoss: Handles both positive and negative samples")
        
    def smooth_BCE(self, eps=0.1):
        """Label smoothing for BCE targets"""
        return 1.0 - 0.5 * eps, 0.5 * eps
    
    def forward(self, predictions, targets):
        """Forward pass that properly handles negative samples"""
        
        # Prepare targets
        targets_tensor = self._prepare_targets(targets)
        
        # Convert predictions
        p = self._prepare_predictions(predictions)
        
        # Adjust for model architecture
        if len(p) != self.nl:
            self._adjust_to_predictions(len(p))
        
        bs = p[0].shape[0]
        loss = torch.zeros(3, device=self.device)  # [box, obj, cls] losses
        
        # Check if we have any positive samples in this batch
        has_positive_samples = targets_tensor.shape[0] > 0
        
        if has_positive_samples:
            # Build targets for positive samples
            tcls, tbox, indices, anchors = self.build_targets(p, targets_tensor)
            logger.debug(f"Batch has {targets_tensor.shape[0]} positive targets")
        else:
            # No positive samples - create empty target lists
            tcls = [torch.zeros(0, dtype=torch.long, device=self.device) for _ in range(len(p))]
            tbox = [torch.zeros(0, 4, device=self.device) for _ in range(len(p))]
            indices = [(torch.zeros(0, dtype=torch.long, device=self.device),) * 4 for _ in range(len(p))]
            anchors = [torch.zeros(0, 2, device=self.device) for _ in range(len(p))]
            logger.debug("Batch has no positive targets (background only)")
        
        # Calculate losses for each detection layer
        for i, pi in enumerate(p):
            if i >= len(indices):
                continue
                
            # Target objectness (all zeros for background, IoU for positive)
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)
            
            if has_positive_samples and len(indices[i]) == 4:
                b, a, gj, gi = indices[i]
                n = b.shape[0]  # number of targets for this layer
                
                if n > 0:
                    # Prediction subset for positive samples
                    pxy, pwh, pobj, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.num_classes), 1)
                    
                    # Regression loss (only for positive samples)
                    pxy = pxy.sigmoid() * 2 - 0.5
                    pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                    pbox = torch.cat((pxy, pwh), 1)
                    
                    # IoU calculation
                    iou = self.bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                    if iou.numel() > 0:
                        loss[0] += (1.0 - iou).mean()  # Box loss
                        
                        # Set positive objectness targets
                        iou = iou.detach().clamp(0).type(tobj.dtype)
                        if self.gr < 1:
                            iou = (1.0 - self.gr) + self.gr * iou
                        tobj[b, a, gj, gi] = iou
                    
                    # Classification loss (only for positive samples)
                    if self.num_classes > 0:
                        t = torch.full_like(pcls, self.cn, device=self.device)
                        if len(tcls[i]) > 0:
                            t[range(n), tcls[i]] = self.cp
                        loss[2] += self.BCEcls(pcls, t)
            
            # Objectness loss (CRITICAL for negative samples)
            # This teaches the model when NO objects are present
            obji = self.BCEobj(pi[..., 4], tobj)
            loss[1] += obji * (self.balance[i] if i < len(self.balance) else 1.0)
            
            # Auto-balance
            if self.autobalance and i < len(self.balance):
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        
        # Auto-balance normalization
        if self.autobalance and len(self.balance) > 1:
            self.balance = [x / self.balance[1] for x in self.balance]
        
        # Apply loss weights
        loss[0] *= self.hyp['box']   # Box loss (zero for negative samples)
        loss[1] *= self.hyp['obj']   # Objectness loss (non-zero for all samples)
        loss[2] *= self.hyp['cls']   # Class loss (zero for negative samples)
        
        # For negative samples, we expect:
        # - Box loss = 0 (no boxes to regress)
        # - Obj loss > 0 (learning to predict no objects)
        # - Cls loss = 0 (no classes to classify)
        
        total_loss = loss.sum() * bs
        
        if not has_positive_samples:
            logger.debug(f"Negative sample batch - Obj loss: {loss[1]:.6f}")
        else:
            logger.debug(f"Positive sample batch - Box: {loss[0]:.6f}, Obj: {loss[1]:.6f}, Cls: {loss[2]:.6f}")
        
        return total_loss, loss.detach()
    
    def _prepare_targets(self, targets):
        """Prepare targets, allowing for empty target batches"""
        if isinstance(targets, dict):
            batch_size = targets['images'].shape[0]
            target_list = []
            
            for i in range(batch_size):
                bboxes = targets['bboxes'][i]
                labels = targets['labels'][i]
                
                # Check for valid targets (non-zero bbox and positive label)
                valid_mask = (bboxes.sum(dim=1) > 0) & (labels > 0)
                bboxes = bboxes[valid_mask]
                labels = labels[valid_mask]
                
                if len(bboxes) > 0:
                    # Image has positive targets
                    img_idx = torch.full((len(bboxes), 1), i, dtype=torch.float32, device=self.device)
                    
                    # Convert to 0-based class indexing
                    if labels.max() > 0:
                        labels = labels - 1
                    labels = torch.clamp(labels, 0, self.num_classes - 1)
                    
                    targets_i = torch.cat([
                        img_idx,
                        labels.float().unsqueeze(1),
                        bboxes
                    ], dim=1)
                    target_list.append(targets_i)
                # If no valid targets, we don't add anything (background image)
            
            if target_list:
                return torch.cat(target_list, dim=0)
            else:
                # All images in batch are background
                return torch.zeros((0, 6), device=self.device)
        
        return targets
    
    def _prepare_predictions(self, predictions):
        """Convert predictions to proper format"""
        p = []
        for i, (cls_score, bbox_pred, objectness) in enumerate(predictions):
            B, _, H, W = cls_score.shape
            
            # Reshape predictions
            cls_score = cls_score.view(B, self.na, self.num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()
            bbox_pred = bbox_pred.view(B, self.na, 4, H, W).permute(0, 1, 3, 4, 2).contiguous()
            objectness = objectness.view(B, self.na, 1, H, W).permute(0, 1, 3, 4, 2).contiguous()
            
            pred = torch.cat([bbox_pred, objectness, cls_score], dim=-1)
            p.append(pred)
        
        return p
    
    def _adjust_to_predictions(self, num_predictions):
        """Adjust loss parameters to match number of prediction scales"""
        if num_predictions != self.nl:
            logger.warning(f"Adjusting nl from {self.nl} to {num_predictions}")
            self.nl = num_predictions
            
            # Adjust balance weights
            if num_predictions == 4:
                self.balance = [8.0, 4.0, 1.0, 0.25]
                # Add extra anchor if needed
                if len(self.anchors) < 4:
                    extra_anchor = torch.tensor([[3, 4], [5, 6], [8, 10]], 
                                              dtype=torch.float32, device=self.device)
                    self.anchors = torch.cat([extra_anchor.unsqueeze(0), self.anchors], dim=0)
            else:
                self.balance = [1.0] * num_predictions
                self.anchors = self.anchors[:num_predictions] if len(self.anchors) > num_predictions else self.anchors
    
    def bbox_iou(self, box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        """Calculate IoU between boxes"""
        # Get coordinates
        if xywh:
            (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
            w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
            b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
            b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        # Intersection
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union
        
        if CIoU or DIoU or GIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
            if CIoU or DIoU:
                c2 = cw ** 2 + ch ** 2 + eps
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
                if CIoU:
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)
                return iou - rho2 / c2
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
        return iou
    
    def build_targets(self, p, targets):
        """Build targets for positive samples only"""
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)
        
        g = 0.5
        off = torch.tensor([
            [0, 0],
            [1, 0], [0, 1], [-1, 0], [0, -1],
        ], device=self.device).float() * g
        
        for i in range(min(self.nl, len(p))):
            anchors = self.anchors[i] if i < len(self.anchors) else self.anchors[-1]
            shape = p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]
            
            t = targets * gain
            if nt:
                r = t[..., 4:6] / anchors[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']
                t = t[j]
                
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            
            bc, gxy, gwh, a = t.chunk(4, 1)
            a, (b, c) = a.long().view(-1), bc.long().T
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)
        
        return tcls, tbox, indices, anch
