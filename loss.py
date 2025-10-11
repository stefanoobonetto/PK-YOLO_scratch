import math
import torch
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class YOLOLoss(nn.Module):
    """YOLO detection loss with Focaler-IoU (PK-YOLO paper). Single-class detection only."""
    
    def __init__(self, model, anchors=None, autobalance=False, img_size=640):
        super().__init__()
        
        device = next(model.parameters()).device
        self.device = device
        self.img_size = img_size
        
        # Loss hyperparameters
        self.hyp = {
            'box': 0.05,
            'obj': 1.0,
            'anchor_t': 4.0,
            'reg_metric': 'focal_ciou',
            'focal_gamma': 1.5,
            'focaler_d': 0.0,
            'focaler_u': 0.95
        }
            
        self.BCEobj = nn.BCEWithLogitsLoss(reduction='none')
        
        # Anchors optimized for Brats dataset ()
        if anchors is None:
            self.anchors = torch.tensor([
                [[16.2, 14.4], [41.1, 33.3], [74.1, 57.6]],       
                [[110.4, 84.0], [146.4, 107.1], [180.3, 132.6]],   
                [[226.0, 129.3], [214.8, 188.8], [278.2, 173.3]] 
            ], dtype=torch.float32).view(3, 3, 2)
        else:
            self.anchors = anchors.to(device)
        
        self.nl = len(self.anchors)
        self.na = self.anchors.shape[1]
        
        # Per-level loss balancing
        self.balance = [4.0, 1.0, 0.25] if self.nl == 3 else [1.0] * self.nl
        self.autobalance = autobalance
            
    def forward(self, predictions, targets):
        """Compute total loss from predictions and targets."""
        targets_tensor = self.prepare_targets(targets)
        p = self.prepare_predictions(predictions)

        # Calculate stride for each detection level
        strides = []
        for pi in p:
            _, _, H, W, _ = pi.shape
            strides.append(self.img_size / max(H, W))
        
        # Scale anchors pixel space --> grid space
        scaled_anchors = [self.anchors[i].to(self.device) / strides[i] for i in range(len(p))]

        if len(p) != self.nl:
            self.adjust_to_predictions(len(p))
        
        loss = torch.zeros(2, device=self.device)  # [box_loss, obj_loss]
        
        has_positive_samples = targets_tensor.shape[0] > 0
        
        if has_positive_samples:
            tbox, indices, anch = self.build_targets(p, targets_tensor, scaled_anchors)
        else:
            tbox = [torch.zeros(0, 4, device=self.device) for _ in range(len(p))]
            indices = [(torch.zeros(0, dtype=torch.long, device=self.device),) * 4 for _ in range(len(p))]
            anch = [torch.zeros(0, 2, device=self.device) for _ in range(len(p))]
        
        # Compute losses for each detection level
        for i, pi in enumerate(p):
            if i >= len(indices):
                continue
                
            b_size, n_anchors, h, w, n_outputs = pi.shape
            tobj = torch.zeros((b_size, n_anchors, h, w), dtype=pi.dtype, device=self.device)
            
            if has_positive_samples and len(indices[i]) == 4:
                b, a, gj, gi = indices[i]
                
                if b.numel() > 0:
                    ps = pi[b, a, gj, gi]
                    
                    # Decode predictions
                    pxy = torch.sigmoid(ps[:, 0:2]) * 2.0 - 0.5
                    pwh = (torch.sigmoid(ps[:, 2:4]) * 2) ** 2 * anch[i]
                    pbox = torch.cat((pxy, pwh), 1)
                    
                    # Compute IoU metrics
                    iou_ciou = self.bbox_iou(pbox, tbox[i], CIoU=True).squeeze(-1).clamp(0.0, 1.0)
                    iou_giou = self.bbox_iou(pbox, tbox[i], GIoU=True).squeeze(-1).clamp(-1.0, 1.0)
                    iou_raw = self.bbox_iou(pbox, tbox[i], CIoU=False).squeeze(-1).clamp(0.0, 1.0)

                    metric = self.hyp.get('reg_metric', 'focal_ciou')
                    gamma = float(self.hyp.get('focal_gamma', 1.5))
                    d = float(self.hyp.get('focaler_d', 0.0))
                    u = float(self.hyp.get('focaler_u', 0.95))

                    # Compute box loss based on selected metric
                    if metric == 'focal_ciou':
                        box_loss = (1.0 - iou_ciou).pow(gamma).mean()
                    elif metric == 'focal_giou':
                        giou01 = (iou_giou + 1.) / 2.0
                        box_loss = (1.0 - giou01).pow(gamma).mean()
                    elif metric == 'giou':
                        box_loss = (1.0 - ((iou_giou + 1.) / 2.0)).mean()
                    elif metric == 'focaler_ciou':
                        iou_focaler = self.focaler_map(iou_raw, d=d, u=u)
                        base = (1.0 - iou_ciou)
                        delta = (iou_raw - iou_focaler)
                        box_loss = (base + delta).mean()
                    elif metric == 'focaler_giou':
                        iou_focaler = self.focaler_map(iou_raw, d=d, u=u)
                        giou01 = (iou_giou + 1.) / 2.0
                        base = (1.0 - giou01)
                        delta = (iou_raw - iou_focaler)
                        box_loss = (base + delta).mean()
                    else:  # 'ciou'
                        box_loss = (1.0 - iou_ciou).mean()
                    
                    # Set objectness targets
                    iou_detached = iou_ciou.detach().clamp(0.0, 1.0)
                    tobj[b, a, gj, gi] = iou_detached.to(tobj.dtype)
                else:
                    box_loss = torch.tensor(0.0, device=self.device)
            else:
                box_loss = torch.tensor(0.0, device=self.device)
            
            # Objectness loss
            obj_loss = self.BCEobj(pi[..., 4], tobj).mean()
            
            # Apply per-level balancing
            loss[0] += box_loss * self.balance[min(i, len(self.balance)-1)]
            loss[1] += obj_loss * self.balance[min(i, len(self.balance)-1)]
            
            # Auto-balance weights
            if self.autobalance:
                objl = float(max(obj_loss.item(), 1e-3))
                self.balance[i] = float(self.balance[i] * 0.9999 + 0.0001 / objl)
        
        # Normalize balance weights
        if self.autobalance and isinstance(self.balance, list) and len(self.balance) > 0:
            b = torch.tensor(self.balance, device=self.device, dtype=torch.float32)
            b = (b / b.mean().clamp_(min=1e-6)).clamp_(0.25, 4.0)
            self.balance = b.detach().cpu().tolist()

        # Apply global loss weights
        loss[0] *= self.hyp['box']
        loss[1] *= self.hyp['obj']
        
        # NaN protection
        for i in range(2):
            if torch.isnan(loss[i]):
                loss[i] = torch.tensor(0.1, device=self.device)
        
        total_loss = loss.sum()
        
        if torch.isnan(total_loss):
            total_loss = torch.tensor(1.0, device=self.device)
            loss = torch.tensor([0.2, 0.8], device=self.device)
        
        return total_loss, loss.detach()
    
    def focaler_map(self, iou: torch.Tensor, d: float = 0.0, u: float = 0.95) -> torch.Tensor:
        """Piecewise-linear remapping for Focaler-IoU (Eq. 14)."""
        eps = 1e-7
        d = float(d)
        u = float(u)
        
        if not (0.0 <= d < u <= 1.0):
            d, u = 0.0, 0.95
            
        mapped = (iou - d) / max(u - d, eps)
        return mapped.clamp_(0.0, 1.0)
    
    def bbox_iou(self, box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        """Calculate IoU between boxes. Supports standard IoU, GIoU, DIoU, and CIoU."""
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
        union = torch.clamp(union, min=eps)
        
        # Standard IoU
        iou = inter / union
        iou = torch.clamp(iou, min=0.0, max=1.0)
        
        # Advanced IoU variants
        if CIoU or DIoU or GIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
            
            if CIoU or DIoU:
                c2 = cw ** 2 + ch ** 2 + eps
                c2 = torch.clamp(c2, min=eps)
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + 
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / c2
                
                if CIoU:
                    v = (4 / (math.pi ** 2)) * torch.pow(
                        torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
                    )
                    with torch.no_grad():
                        alpha = v / (v - iou + 1 + eps)
                    iou = iou - (rho2 + v * alpha)
                else:  # DIoU
                    iou = iou - rho2
            
            if GIoU:
                c_area = cw * ch + eps
                iou = iou - (c_area - union) / c_area

        return iou
   
    def prepare_targets(self, targets):
        """Convert targets to standardized tensor format [N, 5]: (batch_idx, x, y, w, h)."""
        if isinstance(targets, dict):
            batch_size = targets['images'].shape[0]
            target_list = []
            
            for i in range(batch_size):
                bboxes = targets['bboxes'][i]
                
                if torch.isnan(bboxes).any() or torch.isinf(bboxes).any():
                    continue

                valid_mask = bboxes.sum(dim=1) > 0
                valid_bboxes = bboxes[valid_mask]

                if len(valid_bboxes) > 0:
                    # Normalize to [0,1] if in pixel coordinates
                    if (valid_bboxes.max() > 1.0 + 1e-6):
                        valid_bboxes = valid_bboxes / float(self.img_size)

                    img_idx = torch.full((len(valid_bboxes), 1), i, 
                                        dtype=torch.float32, device=self.device)

                    targets_i = torch.cat([img_idx, valid_bboxes], dim=1)
                    target_list.append(targets_i)

            return torch.cat(target_list, dim=0) if target_list else \
                   torch.zeros((0, 5), device=self.device)

        return targets
   
    def prepare_predictions(self, predictions):
        """Convert predictions to [B, na, H, W, 5] format (bbox + objectness)."""
        p = []
        for i, (bbox_pred, objectness) in enumerate(predictions):
            # NaN protection
            if torch.isnan(bbox_pred).any() or torch.isnan(objectness).any():
                bbox_pred = torch.nan_to_num(bbox_pred, nan=0.0, posinf=10.0, neginf=-10.0)
                objectness = torch.nan_to_num(objectness, nan=0.0, posinf=1.0, neginf=-1.0)
            
            B, _, H, W = bbox_pred.shape
            
            # Reshape to [B, na, H, W, C]
            bbox_pred = bbox_pred.view(B, self.na, 4, H, W).permute(0, 1, 3, 4, 2).contiguous()
            objectness = objectness.view(B, self.na, 1, H, W).permute(0, 1, 3, 4, 2).contiguous()
            
            pred = torch.cat([bbox_pred, objectness], dim=-1)
            p.append(pred)
        
        return p
    
    def adjust_to_predictions(self, num_predictions):
        """Adjust balance weights when number of prediction levels changes."""
        if num_predictions != self.nl:
            self.nl = num_predictions
            self.balance = [1.0] * num_predictions
    
    def build_targets(self, p, targets, scaled_anchors):
        """Match ground truth boxes to anchors at each detection level."""
        na = self.na
        nt = targets.shape[0]
        tbox, indices, anch = [], [], []
        device = self.device

        if nt == 0:
            for i in range(len(p)):
                tbox.append(torch.zeros(0, 4, device=device))
                indices.append((torch.zeros(0, dtype=torch.long, device=device),) * 4)
                anch.append(torch.zeros(0, 2, device=device))
            return tbox, indices, anch

        # Replicate targets for each anchor
        gain = torch.ones(6, device=device)
        ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)

        for i in range(len(p)):
            B, na_l, H, W, _ = p[i].shape
            anchors = scaled_anchors[i]

            # Scale targets to grid
            gain[1:5] = torch.tensor([W, H, W, H], device=device)
            t = targets * gain

            if t.numel() == 0:
                tbox.append(torch.zeros(0, 4, device=device))
                indices.append((torch.zeros(0, dtype=torch.long, device=device),) * 4)
                anch.append(torch.zeros(0, 2, device=device))
                continue

            # Anchor matching by aspect ratio
            r = t[:, :, 3:5] / anchors[:, None]
            j = torch.max(r, 1. / (r + 1e-9)).max(2)[0] < self.hyp['anchor_t']
            t = t[j]

            if t.shape[0] == 0:
                tbox.append(torch.zeros(0, 4, device=device))
                indices.append((torch.zeros(0, dtype=torch.long, device=device),) * 4)
                anch.append(torch.zeros(0, 2, device=device))
                continue

            # Extract components
            b, gxy, gwh, a = t[:, 0].long(), t[:, 1:3], t[:, 3:5], t[:, 5].long()
            
            # Grid indices
            gij = gxy.long()
            gi = gij[:, 0].clamp_(0, W - 1)
            gj = gij[:, 1].clamp_(0, H - 1)

            # Target box in grid space
            tbox_i = torch.cat((gxy - gij.float(), gwh), 1)
            
            indices.append((b, a, gj, gi))
            tbox.append(tbox_i)
            anch.append(anchors[a])

        return tbox, indices, anch