import math
import torch
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)

class YOLOLoss(nn.Module):
    
    def __init__(self, model, num_classes=1, anchors=None, autobalance=False, img_size=640):
        super().__init__()
        
        device = next(model.parameters()).device
        self.device = device
        self.num_classes = num_classes
        self.img_size = img_size

        self.hyp = {
            'box': 0.05,
            'cls': 0.3,
            'obj': 1.0,
            'anchor_t': 4.0,
            'fl_gamma': 0.0,
            'cls_pw': 1.0,
            'obj_pw': 1.0,
            'label_smoothing': 0.0,
        }
        
        self.BCEcls = nn.BCEWithLogitsLoss(reduction='none')
        self.BCEobj = nn.BCEWithLogitsLoss(reduction='none')
        
        if anchors is None:
            self.anchors = torch.tensor([
                [[10, 13], [16, 30], [33, 23]],
                [[30, 61], [62, 45], [59, 119]],
                [[116, 90], [156, 198], [373, 326]]
            ], dtype=torch.float32, device=device)
        else:
            self.anchors = anchors.to(device)
        
        self.nl = len(self.anchors)
        self.na = self.anchors.shape[1]
        
        self.balance = [4.0, 1.0, 0.25] if self.nl == 3 else [1.0] * self.nl
        self.autobalance = autobalance
        self.gr = 1.0
            
    def forward(self, predictions, targets):        
        targets_tensor = self._prepare_targets(targets)
        p = self._prepare_predictions(predictions)
        
        if not hasattr(self, '_debug_calls'):
            self._debug_calls = 0
        if self._debug_calls < 6:
            try:
                import logging
                logging.getLogger(__name__).info(
                    f"[YOLOLoss] pred levels: {[tuple(pi.shape) for pi in p]} | targets: {tuple(targets_tensor.shape)}"
                )
            except Exception:
                pass
        self._debug_calls += 1

        # p[i].shape = (B, na, H, W, n_outputs)
        grid_hw = [pi.shape[2:4] for pi in p]            # [(H,W), ...]
        img_size = 640.0
        strides = [img_size / float(w) for (_, w) in grid_hw]
        scaled_anchors = [
            self.anchors[i].to(self.device) / float(strides[i])
            for i in range(min(self.nl, len(p)))
        ]

        if len(p) != self.nl:
            self._adjust_to_predictions(len(p))
        
        bs = p[0].shape[0]
        loss = torch.zeros(3, device=self.device)
        
        has_positive_samples = targets_tensor.shape[0] > 0
        
        if has_positive_samples:
            tcls, tbox, indices, anch = self.build_targets(p, targets_tensor, scaled_anchors)
            npos = sum(len(idx[0]) for idx in indices)
            try:
                import logging
                logging.getLogger(__name__).info(
                    f"[YOLOLoss] assignments per level: {[len(idx[0]) for idx in indices]} (total {npos})"
                )
            except Exception:
                pass
            if npos == 0:
                logger.warning("YOLOLoss: 0 positive matches this batch")
        else:
            tcls = [torch.zeros(0, dtype=torch.long, device=self.device) for _ in range(len(p))]
            tbox = [torch.zeros(0, 4, device=self.device) for _ in range(len(p))]
            indices = [(torch.zeros(0, dtype=torch.long, device=self.device),) * 4 for _ in range(len(p))]
            anch = [torch.zeros(0, 2, device=self.device) for _ in range(len(p))]
        
        for i, pi in enumerate(p):
            if i >= len(indices):
                continue
                
            b_size, n_anchors, h, w, n_outputs = pi.shape
            tobj = torch.zeros((b_size, n_anchors, h, w), dtype=pi.dtype, device=self.device)
            
            if has_positive_samples and len(indices[i]) == 4:
                b, a, gj, gi = indices[i]
                n = b.shape[0]
                
                if n > 0:
                    ps = pi[b, a, gj, gi]
                    
                    pxy = ps[:, 0:2]
                    pwh = ps[:, 2:4]
                    pobj = ps[:, 4:5]
                    pcls = ps[:, 5:] if self.num_classes > 0 else torch.zeros_like(pobj)
                    
                    pxy = torch.sigmoid(ps[:, 0:2]) * 2.0 - 0.5
                    pwh = (torch.sigmoid(ps[:, 2:4]) * 2) ** 2 * anch[i]  
                    pbox = torch.cat((pxy, pwh), 1)    
                    
                    iou = self.bbox_iou(pbox, tbox[i], CIoU=True).squeeze(-1)
                    iou = torch.clamp(iou, min=0.0, max=1.0)
                    
                    if iou.numel() > 0 and not torch.isnan(iou).any():
                        box_loss = (1.0 - iou).mean()
                        
                        if not torch.isnan(box_loss):
                            loss[0] += box_loss
                        
                        iou_detached = iou.detach().clamp(0, 1).type(tobj.dtype)
                        if self.gr < 1:
                            iou_detached = (1.0 - self.gr) + self.gr * iou_detached
                        tobj[b, a, gj, gi] = iou_detached
                    
                    if self.num_classes > 0 and pcls.numel() > 0:
                        t = torch.zeros_like(pcls, device=self.device)
                        if len(tcls[i]) > 0:
                            t[range(n), tcls[i]] = 1.0
                        
                        cls_loss = self.BCEcls(pcls, t)
                        cls_loss = torch.clamp(cls_loss, min=0.0, max=100.0)
                        cls_loss_mean = cls_loss.mean()
                        
                        if not torch.isnan(cls_loss_mean):
                            loss[2] += cls_loss_mean
            
            pi_obj = pi[..., 4]
            obj_loss = self.BCEobj(pi_obj, tobj)
            obj_loss = torch.clamp(obj_loss, min=0.0, max=100.0)
            
            if not torch.isnan(obj_loss).any():
                obj_loss_weighted = obj_loss.mean() * (self.balance[i] if i < len(self.balance) else 1.0)
                loss[1] += obj_loss_weighted
            else:
                loss[1] += torch.tensor(0.1, device=self.device)
        
        loss[0] *= self.hyp['box']
        loss[1] *= self.hyp['obj']
        loss[2] *= self.hyp['cls']
        
        for i in range(3):
            if torch.isnan(loss[i]):
                loss[i] = torch.tensor(0.1, device=self.device)
        
        total_loss = loss.sum() * bs
        
        if torch.isnan(total_loss):
            total_loss = torch.tensor(1.0, device=self.device)
            loss = torch.tensor([0.1, 0.8, 0.1], device=self.device)
        
        return total_loss, loss.detach()
    
    def bbox_iou(self, box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):        
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

        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        union = w1 * h1 + w2 * h2 - inter + eps
        union = torch.clamp(union, min=eps)
        
        iou = inter / union
        iou = torch.clamp(iou, min=0.0, max=1.0)
        
        if CIoU or DIoU or GIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
            
            if CIoU or DIoU:
                c2 = cw ** 2 + ch ** 2 + eps
                c2 = torch.clamp(c2, min=eps)
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
                
                if CIoU:
                    w2_safe = torch.clamp(w2, min=eps)
                    h2_safe = torch.clamp(h2, min=eps)
                    w1_safe = torch.clamp(w1, min=eps)
                    h1_safe = torch.clamp(h1, min=eps)
                    
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2_safe / h2_safe) - torch.atan(w1_safe / h1_safe), 2)
                    v = torch.clamp(v, min=0.0, max=4.0)
                    
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                        alpha = torch.clamp(alpha, min=0.0, max=1.0)
                    
                    ciou = iou - (rho2 / c2 + v * alpha)
                    return torch.clamp(ciou, min=-1.0, max=1.0)
                
                diou = iou - rho2 / c2
                return torch.clamp(diou, min=-1.0, max=1.0)
            
            c_area = cw * ch + eps
            giou = iou - (c_area - union) / c_area
            return torch.clamp(giou, min=-1.0, max=1.0)
        
        return iou
    
    def _prepare_targets(self, targets):
        if isinstance(targets, dict):
            batch_size = targets['images'].shape[0]
            target_list = []
            
            for i in range(batch_size):
                bboxes = targets['bboxes'][i]
                labels = targets['labels'][i]
                
                if torch.isnan(bboxes).any() or torch.isinf(bboxes).any():
                    continue
                
                valid_mask = (bboxes.sum(dim=1) > 0) & (labels >= 0)
                valid_bboxes = bboxes[valid_mask]
                valid_labels = labels[valid_mask]
                
                if len(valid_bboxes) > 0:
                    img_idx = torch.full((len(valid_bboxes), 1), i, dtype=torch.float32, device=self.device)
                    valid_labels = torch.clamp(valid_labels, 0, self.num_classes - 1)
                    
                    targets_i = torch.cat([
                        img_idx,
                        valid_labels.float().unsqueeze(1),
                        valid_bboxes
                    ], dim=1)
                    target_list.append(targets_i)
            
            if target_list:
                return torch.cat(target_list, dim=0)
            else:
                return torch.zeros((0, 6), device=self.device)
        
        return targets
    
    def _prepare_predictions(self, predictions):
        p = []
        for i, (cls_score, bbox_pred, objectness) in enumerate(predictions):
            if torch.isnan(cls_score).any() or torch.isnan(bbox_pred).any() or torch.isnan(objectness).any():
                cls_score = torch.nan_to_num(cls_score, nan=0.0, posinf=1.0, neginf=-1.0)
                bbox_pred = torch.nan_to_num(bbox_pred, nan=0.0, posinf=10.0, neginf=-10.0)
                objectness = torch.nan_to_num(objectness, nan=0.0, posinf=1.0, neginf=-1.0)
            
            B, _, H, W = cls_score.shape
            
            cls_score = cls_score.view(B, self.na, self.num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()
            bbox_pred = bbox_pred.view(B, self.na, 4, H, W).permute(0, 1, 3, 4, 2).contiguous()
            objectness = objectness.view(B, self.na, 1, H, W).permute(0, 1, 3, 4, 2).contiguous()
            
            pred = torch.cat([bbox_pred, objectness, cls_score], dim=-1)
            p.append(pred)
        
        return p
    
    def _adjust_to_predictions(self, num_predictions):
        if num_predictions != self.nl:
            self.nl = num_predictions
            self.balance = [1.0] * num_predictions
    
    def build_targets(self, p, targets, scaled_anchors):
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
            anchors = scaled_anchors[i] if i < len(scaled_anchors) else scaled_anchors[-1]
            shape = p[i].shape
            gain[2:6] = torch.tensor(shape, device=self.device)[[3, 2, 3, 2]]

            t = targets * gain
            if nt:
                t[:, :, 2:6] = torch.clamp(t[:, :, 2:6], min=0.0, max=float(max(shape[2:4])))
                
                r = t[..., 4:6] / anchors[:, None]
                j = torch.max(r, 1 / (r + 1e-16)).max(2)[0] < self.hyp['anchor_t']
                t = t[j]
                
                if t.shape[0] > 0:
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
            else:
                t = targets[0]
                offsets = 0
            
            if t.shape[0] > 0:
                bc, gxy, gwh, a = t.chunk(4, 1)
                a, (b, c) = a.long().view(-1), bc.long().T
                gij = (gxy - offsets).long()
                gi, gj = gij.T
                
                gi = gi.clamp_(0, shape[3] - 1)
                gj = gj.clamp_(0, shape[2] - 1)
                
                indices.append((b, a, gj, gi))
                tbox.append(torch.cat((gxy - gij, gwh), 1))
                anch.append(anchors[a])
                tcls.append(c)
            else:
                indices.append((torch.zeros(0, dtype=torch.long, device=self.device),) * 4)
                tbox.append(torch.zeros(0, 4, device=self.device))
                anch.append(torch.zeros(0, 2, device=self.device))
                tcls.append(torch.zeros(0, dtype=torch.long, device=self.device))
        
        return tcls, tbox, indices, anch