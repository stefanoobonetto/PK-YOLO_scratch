import torch
import torch.nn as nn
import math

class YOLOLoss(nn.Module):
    """Optimized YOLO loss for small tumor detection."""
    
    def __init__(self, model, anchors=None, img_size=640):
        super().__init__()
        
        self.device = next(model.parameters()).device
        self.img_size = img_size
        
        # Optimized hyperparameters for small objects
        self.hyp = {
            'box': 0.1,          # Increased for better localization
            'obj': 1.0,          # Standard objectness weight
            'obj_pw': 1.5,       # Positive weight for objectness (handle imbalance)
            'anchor_t': 3.0,     # Stricter anchor matching for small objects
            'fl_gamma': 1.5,     # Focal loss gamma
        }
        
        # Focal loss for objectness (handles imbalance)
        self.BCEobj = nn.BCEWithLogitsLoss(reduction='none')
        
        # Optimized anchors for small tumors
        if anchors is None:
            self.anchors = torch.tensor([
                # P2 - tiny tumors
                [[4, 4], [8, 7], [13, 11]],
                # P3 - small tumors  
                [[19, 17], [28, 24], [41, 35]],
                # P4 - medium tumors
                [[59, 52], [87, 74], [122, 98]],
                # P5 - large tumors
                [[165, 141], [234, 187], [340, 280]]
            ], dtype=torch.float32).to(self.device)
        else:
            self.anchors = anchors.to(self.device)
        
        self.nl = len(self.anchors)  # number of detection layers
        self.na = self.anchors.shape[1]  # number of anchors per layer
        
        # Scale balancing (emphasize high-res layers for small objects)
        self.balance = [8.0, 4.0, 1.0, 0.4] if self.nl == 4 else [4.0, 1.0, 0.4]
    
    def forward(self, predictions, targets):
        """Compute loss."""
        device = self.device
        
        # Prepare targets
        targets_tensor = self._prepare_targets(targets)
        
        # Prepare predictions  
        p = self._prepare_predictions(predictions)
        
        # Build targets for each layer
        tbox, indices, anch = self._build_targets(p, targets_tensor)
        
        # Loss components
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        
        # Compute loss for each detection layer
        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 4])
            
            n = len(b)
            if n:
                # Get predictions for positive samples
                ps = pi[b, a, gj, gi]
                
                # Decode box predictions
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anch[i]
                pbox = torch.cat((pxy, pwh), 1)
                
                # Calculate IoU
                iou = self._bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                
                # Box loss (Focal-EIoU for small objects)
                lbox += ((1.0 - iou) ** self.hyp['fl_gamma']).mean() * self.balance[i]
                
                # Objectness targets (use IoU as soft label)
                tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)
            
            # Objectness loss with focal weighting
            obji = self.BCEobj(pi[..., 4], tobj)
            
            # Apply positive weight to handle imbalance
            obji[tobj > 0] *= self.hyp['obj_pw']
            
            # Focal loss modulation
            p_obj = pi[..., 4].sigmoid()
            focal_weight = tobj * (1.0 - p_obj) ** 2 + (1.0 - tobj) * p_obj ** 2
            
            lobj += (obji * focal_weight).mean() * self.balance[i]
        
        # Apply loss weights
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        
        loss = lbox + lobj
        return loss, torch.cat([lbox.detach(), lobj.detach()])
    
    def _prepare_targets(self, targets):
        """Convert targets to tensor format."""
        if isinstance(targets, dict):
            device = self.device
            batch_size = targets['images'].shape[0]
            target_list = []
            
            for i in range(batch_size):
                bboxes = targets['bboxes'][i]
                valid = bboxes.sum(dim=1) > 0
                
                if valid.any():
                    valid_boxes = bboxes[valid]
                    # Normalize if needed
                    if valid_boxes.max() > 1.01:
                        valid_boxes = valid_boxes / self.img_size
                    
                    batch_idx = torch.full((valid.sum(), 1), i, device=device)
                    target_list.append(torch.cat([batch_idx, valid_boxes], 1))
            
            return torch.cat(target_list, 0) if target_list else torch.zeros((0, 5), device=device)
        return targets
    
    def _prepare_predictions(self, predictions):
        """Reshape predictions to [B, na, H, W, 5] format."""
        p = []
        for bbox_pred, obj_pred in predictions:
            B, _, H, W = bbox_pred.shape
            
            bbox_pred = bbox_pred.view(B, self.na, 4, H, W).permute(0, 1, 3, 4, 2)
            obj_pred = obj_pred.view(B, self.na, 1, H, W).permute(0, 1, 3, 4, 2)
            
            p.append(torch.cat([bbox_pred, obj_pred], -1))
        return p
    
    def _build_targets(self, p, targets):
        """Match targets to anchors."""
        device = self.device
        nt = targets.shape[0]
        tbox, indices, anch = [], [], []
        
        if nt == 0:
            for i in range(len(p)):
                tbox.append(torch.zeros(0, 4, device=device))
                indices.append((torch.zeros(0, dtype=torch.long, device=device),) * 4)
                anch.append(torch.zeros(0, 2, device=device))
            return tbox, indices, anch
        
        # Get strides for each detection layer
        strides = []
        for pi in p:
            _, _, H, W, _ = pi.shape
            strides.append(self.img_size / H)
        
        # Scale anchors to grid space
        scaled_anchors = []
        for i in range(len(p)):
            scaled_anchors.append(self.anchors[i] / strides[i])
        
        # Prepare targets for all anchors
        gain = torch.ones(6, device=device)
        ai = torch.arange(self.na, device=device).float().view(self.na, 1)
        targets = torch.cat((targets.repeat(self.na, 1, 1), ai[:, :, None]), 2)
        
        for i, pi in enumerate(p):
            anchors = scaled_anchors[i].to(device)
            gain[1:5] = torch.tensor(pi.shape)[[3, 2, 3, 2]]
            
            # Scale targets to grid space
            t = targets * gain
            
            if nt:
                # Match targets to anchors
                r = t[:, :, 3:5] / anchors[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']
                t = t[j]
                
                # Extract components
                b, gxy = t[:, 0].long(), t[:, 1:3]
                gwh, a = t[:, 3:5], t[:, 5].long()
                
                # Grid indices
                gi, gj = gxy.long().T
                gi = gi.clamp_(0, gain[3].long() - 1)
                gj = gj.clamp_(0, gain[2].long() - 1)
                
                # Append
                indices.append((b, a, gj, gi))
                tbox.append(torch.cat((gxy - gi.float().unsqueeze(1), gwh), 1))
                anch.append(anchors[a])
            else:
                indices.append((torch.zeros(0, dtype=torch.long, device=device),) * 4)
                tbox.append(torch.zeros(0, 4, device=device))
                anch.append(torch.zeros(0, 2, device=device))
        
        return tbox, indices, anch
    
    def _bbox_iou(self, box1, box2, xywh=True, CIoU=False):
        """Calculate IoU with CIoU support."""
        eps = 1e-7
        
        if xywh:
            # Convert to xyxy
            b1_x1 = box1[..., 0] - box1[..., 2] / 2
            b1_y1 = box1[..., 1] - box1[..., 3] / 2
            b1_x2 = box1[..., 0] + box1[..., 2] / 2
            b1_y2 = box1[..., 1] + box1[..., 3] / 2
            
            b2_x1 = box2[..., 0] - box2[..., 2] / 2
            b2_y1 = box2[..., 1] - box2[..., 3] / 2
            b2_x2 = box2[..., 0] + box2[..., 2] / 2
            b2_y2 = box2[..., 1] + box2[..., 3] / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        
        # Intersection
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
                (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
        
        # Union
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        
        # IoU
        iou = inter / union
        
        if CIoU:
            # Complete IoU
            cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
            ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
            c2 = cw ** 2 + ch ** 2 + eps
            
            # Center distance
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            
            # Aspect ratio
            v = (4 / math.pi ** 2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))).pow(2)
            with torch.no_grad():
                alpha = v / (1 - iou + v + eps)
            
            return iou - (rho2 / c2 + v * alpha)
        
        return iou