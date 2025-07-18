"""
Complete Multimodal PK-YOLO Loss Implementation
Adapted from PK-YOLO for 4-Channel BraTS2020 Dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List


def smooth_BCE(eps=0.1):
    """Label smoothing for BCE targets"""
    return 1.0 - 0.5 * eps, 0.5 * eps


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate IoU between two sets of boxes
    Args:
        box1: (N, 4) boxes in format [x, y, w, h] or [x1, y1, x2, y2]
        box2: (M, 4) boxes in format [x, y, w, h] or [x1, y1, x2, y2]
        xywh: If True, boxes are in [x, y, w, h] format, else [x1, y1, x2, y2]
        GIoU, DIoU, CIoU: Different IoU variants
    Returns:
        IoU values
    """
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
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
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
    Complete YOLO Loss Function for Multimodal Brain Tumor Detection
    Adapted from PK-YOLO for 4-channel BraTS2020 dataset
    """
    
    def __init__(self, model, num_classes=1, anchors=None, autobalance=False):
        super().__init__()
        
        device = next(model.parameters()).device
        self.device = device
        self.num_classes = num_classes
        
        # Hyperparameters for brain tumor detection
        self.hyp = {
            'box': 0.05,           # Box loss weight
            'cls': 0.5,            # Class loss weight  
            'obj': 1.0,            # Object loss weight
            'anchor_t': 4.0,       # Anchor matching threshold
            'fl_gamma': 0.0,       # Focal loss gamma (0 = no focal loss)
            'cls_pw': 1.0,         # Class positive weight
            'obj_pw': 1.0,         # Object positive weight
            'label_smoothing': 0.0, # Label smoothing epsilon
        }
        
        # Loss functions
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['obj_pw']], device=device))
        
        # Label smoothing
        self.cp, self.cn = smooth_BCE(eps=self.hyp.get('label_smoothing', 0.0))
        
        # Focal loss for hard examples (important for small brain tumors)
        g = self.hyp['fl_gamma']
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        
        self.BCEcls = BCEcls
        self.BCEobj = BCEobj
        
        # Model properties
        self.nl = 3  # Number of detection layers (P3, P4, P5)
        self.na = 3  # Number of anchors per grid
        
        # Default anchors for different scales (adjusted for brain tumor sizes)
        if anchors is None:
            self.anchors = torch.tensor([
                [[10, 13], [16, 30], [33, 23]],      # P3/8 - small tumors
                [[30, 61], [62, 45], [59, 119]],     # P4/16 - medium tumors  
                [[116, 90], [156, 198], [373, 326]]  # P5/32 - large tumors
            ], dtype=torch.float32, device=device)
        else:
            self.anchors = anchors.to(device)
            
        # Balance weights for different scales
        self.balance = [4.0, 1.0, 0.4]  # P3, P4, P5
        self.autobalance = autobalance
        self.gr = 1.0  # IoU ratio
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: List of tensors [(cls, bbox, obj), ...] for each scale
            targets: Ground truth targets in format [img_idx, class, x, y, w, h]
        Returns:
            Total loss and individual loss components
        """
        device = targets.device
        
        # Prepare targets in the expected format
        if isinstance(targets, dict):
            # Convert from dataloader format to YOLO format
            batch_size = targets['images'].shape[0]
            target_list = []
            
            for i in range(batch_size):
                bboxes = targets['bboxes'][i]  # [num_boxes, 4] in format [x, y, w, h]
                labels = targets['labels'][i]  # [num_boxes]
                
                # Filter out empty boxes (all zeros)
                valid_mask = (bboxes.sum(dim=1) > 0)
                bboxes = bboxes[valid_mask]
                labels = labels[valid_mask]
                
                if len(bboxes) > 0:
                    # Create image index column
                    img_idx = torch.full((len(bboxes), 1), i, dtype=torch.float32, device=self.device)
                    
                    # Combine: [img_idx, class, x, y, w, h]
                    targets_i = torch.cat([
                        img_idx,
                        labels.float().unsqueeze(1),
                        bboxes
                    ], dim=1)
                    target_list.append(targets_i)
            
            if target_list:
                targets = torch.cat(target_list, dim=0)
            else:
                targets = torch.zeros((0, 6), device=self.device)
        
        # Convert predictions to proper format
        p = []
        for i, (cls_score, bbox_pred, objectness) in enumerate(predictions):
            # Reshape and concatenate to get [batch, anchors, grid_h, grid_w, predictions]
            B, _, H, W = cls_score.shape
            
            # Reshape predictions
            cls_score = cls_score.view(B, self.na, self.num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()
            bbox_pred = bbox_pred.view(B, self.na, 4, H, W).permute(0, 1, 3, 4, 2).contiguous()  
            objectness = objectness.view(B, self.na, 1, H, W).permute(0, 1, 3, 4, 2).contiguous()
            
            # Concatenate all predictions
            pred = torch.cat([bbox_pred, objectness, cls_score], dim=-1)  # [B, na, H, W, 5+nc]
            p.append(pred)
        
        bs = p[0].shape[0]  # batch size
        loss = torch.zeros(3, device=device)  # [box, obj, cls] losses
        
        # Build targets
        tcls, tbox, indices = self.build_targets(p, targets)
        
        # Calculate losses for each detection layer
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, gj, gi = indices[i]  # image, grid_y, grid_x indices
            tobj = torch.zeros((pi.shape[0], pi.shape[2], pi.shape[3]), 
                             dtype=pi.dtype, device=device)  # target objectness
            
            n_labels = b.shape[0]  # number of targets
            if n_labels:
                # Select predictions corresponding to targets
                ps = pi[b, :, gj, gi]  # prediction subset [n_labels, na, 5+nc]
                
                # Regression (box coordinates)
                pxy = ps[:, :, 0:2].sigmoid() * 1.6 - 0.3  # xy offset
                pwh = (0.2 + ps[:, :, 2:4].sigmoid() * 4.8) * self.anchors[i]  # wh
                pbox = torch.cat((pxy, pwh), 2)  # predicted box
                
                # Calculate IoU
                selected_tbox = tbox[i]
                iou = bbox_iou(pbox, selected_tbox, xywh=True, CIoU=True).squeeze()
                loss[0] += (1.0 - iou).mean()  # Box loss
                
                # Objectness targets
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, gj, gi] = iou  # IoU ratio
                
                # Classification loss
                if self.num_classes > 1:
                    t = torch.full_like(ps[:, :, 5:], self.cn, device=device)
                    t[range(n_labels), :, tcls[i]] = self.cp
                    loss[2] += self.BCEcls(ps[:, :, 5:], t)  # Class loss
            
            # Objectness loss
            obji = self.BCEobj(pi[:, :, :, :, 4], tobj)
            loss[1] += obji * self.balance[i]  # Weighted by scale
            
            # Auto-balance the objectness loss weights
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        
        # Apply loss weights
        if self.autobalance:
            self.balance = [x / self.balance[1] for x in self.balance]  # Normalize to P4
        
        loss[0] *= self.hyp['box']   # Box loss weight
        loss[1] *= self.hyp['obj']   # Objectness loss weight  
        loss[2] *= self.hyp['cls']   # Class loss weight
        
        return loss.sum() * bs, loss.detach()
    
    def build_targets(self, p, targets):
        """
        Build targets for loss computation
        Args:
            p: Predictions list [layer1, layer2, layer3]
            targets: Ground truth [img_idx, class, x, y, w, h] normalized
        Returns:
            tcls: Target classes for each layer
            tbox: Target boxes for each layer  
            indices: Target indices (image, grid_y, grid_x) for each layer
        """
        nt = targets.shape[0]  # number of targets
        tcls, tbox, indices = [], [], []
        gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
        
        # Anchor indices
        ai = torch.arange(self.na, device=targets.device).float().view(self.na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(self.na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        
        g = 0.3  # bias for grid assignment
        off = torch.tensor([[0, 0],
                           [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                           ], device=targets.device).float()  # offsets
        
        for i in range(self.nl):  # for each detection layer
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain
            
            # Match targets to anchors
            t = targets * gain  # shape(3*na, n, 7)
            if nt:
                # Anchor matching
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                t = t[j]  # filter
                
                # Offsets for better grid assignment
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            
            # Define targets
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices
            
            # Append
            indices.append((b, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            tcls.append(c)  # class
            
        return tcls, tbox, indices