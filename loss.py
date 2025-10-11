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
        
        # ------------------------------
        # Hyperparameters (default)
        # ------------------------------
        self.hyp = {
            'box': 0.05,           # Box regression weight
            'cls': 0.5,            # INCREASED from 0.3 - critical for tumor detection
            'obj': 1.0,            # Objectness weight
            'anchor_t': 4.0,       # Anchor threshold
            'fl_gamma': 1.5,       # Add focal loss for hard negatives
            'cls_pw': 1.0,
            'obj_pw': 1.0,
            'label_smoothing': 0.0,
            'reg_metric': 'focal_ciou',  # Better for medical imaging
            'focal_gamma': 1.5,          # Focal gamma for regression
            # >>> NEW: Focaler-IoU thresholds
            'focaler_d': 0.0,            # lower threshold d
            'focaler_u': 0.95            # upper threshold u
        }
            
        self.BCEcls = nn.BCEWithLogitsLoss(reduction='none')
        self.BCEobj = nn.BCEWithLogitsLoss(reduction='none')
        
        if anchors is None:
            self.anchors = torch.tensor([
                [[16.2, 14.4], [41.1, 33.3], [74.1, 57.6]],      # P3 (80x80) - small tumors
                [[110.4, 84.0], [146.4, 107.1], [180.3, 132.6]], # P4 (40x40) - medium tumors  
                [[226.0, 129.3], [214.8, 188.8], [278.2, 173.3]] # P5 (20x20) - large tumors
            ], dtype=torch.float32).view(3, 3, 2)
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
        # if self._debug_calls < 6:
        #     try:
        #         import logging
        #         logging.getLogger(__name__).info(
        #             f"[YOLOLoss] pred levels: {[tuple(pi.shape) for pi in p]} | targets: {tuple(targets_tensor.shape)}"
        #         )
        #     except Exception:
        #         pass
        self._debug_calls += 1

        # p[i].shape = (B, na, H, W, n_outputs)
        strides = []
        for pi in p:
            _, _, H, W, _ = pi.shape
            strides.append(self.img_size / max(H, W))  # 640 / H (H==W)

        scaled_anchors = [self.anchors[i].to(self.device) / strides[i] for i in range(len(p))]

        if len(p) != self.nl:
            self._adjust_to_predictions(len(p))
        
        bs = p[0].shape[0]
        loss = torch.zeros(3, device=self.device)
        
        has_positive_samples = targets_tensor.shape[0] > 0
        
        if has_positive_samples:
            tcls, tbox, indices, anch = self.build_targets(p, targets_tensor, scaled_anchors)
            npos = sum(x.shape[0] for x in tbox)
            if self._debug_calls < 3:
                logger.info(f"[YOLOLoss] matched positives in batch: {npos}")
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
                
                if b.numel() > 0:
                    # gather predictions at positive indices
                    ps = pi[b, a, gj, gi]  # [npos, 5+nc]
                    
                    # decode local offsets (xy) and anchor-scaled wh in grid space
                    pxy = torch.sigmoid(ps[:, 0:2]) * 2.0 - 0.5
                    pwh = (torch.sigmoid(ps[:, 2:4]) * 2) ** 2 * anch[i]  
                    pbox = torch.cat((pxy, pwh), 1)    # [npos, 4] (xywh, grid space)
                    
                    # IoU metrics
                    iou_ciou = self.bbox_iou(pbox, tbox[i], CIoU=True).squeeze(-1).clamp(0.0, 1.0)
                    iou_giou = self.bbox_iou(pbox, tbox[i], GIoU=True).squeeze(-1).clamp(-1.0, 1.0)

                    metric = self.hyp.get('reg_metric', 'focal_ciou')
                    gamma  = float(self.hyp.get('focal_gamma', 1.5))

                    # >>> NEW: base IoU (senza termini addizionali) per la mappatura Focaler
                    iou_raw = self.bbox_iou(pbox, tbox[i], CIoU=False).squeeze(-1).clamp(0.0, 1.0)
                    d = float(self.hyp.get('focaler_d', 0.0))
                    u = float(self.hyp.get('focaler_u', 0.95))

                    if metric == 'focal_ciou':
                        # versione preesistente: (1 - CIoU)^gamma
                        box_loss = (1.0 - iou_ciou).pow(gamma).mean()
                    elif metric == 'focal_giou':
                        giou01   = (iou_giou + 1.) / 2.0
                        box_loss = (1.0 - giou01).pow(gamma).mean()
                    elif metric == 'giou':
                        box_loss = (1.0 - ((iou_giou + 1.) / 2.0)).mean()
                    elif metric == 'focaler_ciou':
                        # Eq. 14–20: L_focaler-ciou = L_ciou + IoU - IoU_focaler
                        iou_focaler = self.focaler_map(iou_raw, d=d, u=u)
                        base = (1.0 - iou_ciou)       # L_CIoU
                        delta = (iou_raw - iou_focaler)
                        box_loss = (base + delta).mean()
                    elif metric == 'focaler_giou':
                        # Variante su GIoU
                        iou_focaler = self.focaler_map(iou_raw, d=d, u=u)
                        giou01 = (iou_giou + 1.) / 2.0
                        base = (1.0 - giou01)         # L_GIoU in [0,2] -> normalizzato
                        delta = (iou_raw - iou_focaler)
                        box_loss = (base + delta).mean()
                    else:  # 'ciou' standard
                        box_loss = (1.0 - iou_ciou).mean()
                    
                    # objectness target = IoU (senza Focaler) per pp. YOLO v5/v9
                    iou_detached = iou_ciou.detach().clamp(0.0, 1.0)
                    tobj[b, a, gj, gi] = iou_detached.to(tobj.dtype)
                    
                    # class loss (se num_classes>0)
                    if self.num_classes > 0:
                        t = torch.zeros_like(ps[:, 5:], device=self.device)
                        # tutte GT sono della stessa classe (una sola): pos = 1.0
                        if t.numel() > 0:
                            t[:] = 1.0
                        cls_loss = self.BCEcls(ps[:, 5:], t).mean()
                    else:
                        cls_loss = torch.tensor(0.0, device=self.device)

                else:
                    box_loss = torch.tensor(0.0, device=self.device)
                    cls_loss = torch.tensor(0.0, device=self.device)
            else:
                box_loss = torch.tensor(0.0, device=self.device)
                cls_loss = torch.tensor(0.0, device=self.device)
            
            # objectness loss
            obj_loss = self.BCEobj(pi[..., 4], tobj).mean()
            
            # per-lvl weighting (come v5/v9)
            loss[0] += box_loss * self.balance[min(i, len(self.balance)-1)]
            loss[1] += obj_loss * self.balance[min(i, len(self.balance)-1)]
            loss[2] += cls_loss * self.balance[min(i, len(self.balance)-1)]
            
            if self.autobalance:
                # Stabilize autobalance to avoid runaway weights when obj_loss -> 0
                objl = float(max(obj_loss.item(), 1e-3))  # clamp to avoid blow-up
                self.balance[i] = float(self.balance[i] * 0.9999 + 0.0001 / objl)
        
        # renormalize balance to keep mean ~1 and clamp to [0.25, 4.0]
        if self.autobalance and isinstance(self.balance, list) and len(self.balance) > 0:
            b = torch.tensor(self.balance, device=self.device, dtype=torch.float32)
            b = (b / b.mean().clamp_(min=1e-6)).clamp_(0.25, 4.0)
            self.balance = b.detach().cpu().tolist()

        # applica pesi globali
        loss[0] *= self.hyp['box']
        loss[1] *= self.hyp['obj']
        loss[2] *= self.hyp['cls']
        
        for i in range(3):
            if torch.isnan(loss[i]):
                loss[i] = torch.tensor(0.1, device=self.device)
        
        total_loss = loss.sum()
        
        if torch.isnan(total_loss):
            total_loss = torch.tensor(1.0, device=self.device)
            loss = torch.tensor([0.1, 0.8, 0.1], device=self.device)
        
        return total_loss, loss.detach()
    
    # ---------- NEW: Focaler mapping ----------
    def focaler_map(self, iou: torch.Tensor, d: float = 0.0, u: float = 0.95) -> torch.Tensor:
        """Piecewise-linear remapping used by Focaler-IoU (Eq. 14).
        Maps IoU in [0,1] to [0,1] with two thresholds d < u.
        """
        eps = 1e-7
        d = float(d); u = float(u)
        if not (0.0 <= d < u <= 1.0):
            d, u = 0.0, 0.95  # fallback sicuro
        mapped = (iou - d) / max(u - d, eps)
        return mapped.clamp_(0.0, 1.0)
    
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
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / c2
                if CIoU:
                    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou + 1 + eps)
                    iou = iou - (rho2 + v * alpha)
                else:
                    iou = iou - rho2
            
            if GIoU:
                c_area = cw * ch + eps
                iou = iou - (c_area - union) / c_area

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
                    #    Normalizza solo se arrivano in pixel (valori > 1)
                    if (valid_bboxes.max() > 1.0 + 1e-6):
                        valid_bboxes = valid_bboxes / float(self.img_size)

                    img_idx = torch.full((len(valid_bboxes), 1), i, dtype=torch.float32, device=self.device)
                    valid_labels = torch.clamp(valid_labels, 0, self.num_classes - 1).view(-1, 1).float()

                    targets_i = torch.cat([img_idx, valid_labels, valid_bboxes], dim=1)  # [N,6]: b,c,x,y,w,h
                    target_list.append(targets_i)

            return torch.cat(target_list, dim=0) if target_list else torch.zeros((0, 6), device=self.device)

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
        """
        Versione semplice e robusta (senza offsets) per evitare mismatch di shape.
        p: list of levels, each (B, na, H, W, C)
        targets: [nt,6] -> (batch_idx, class, x, y, w, h) normalizzati [0,1]
        scaled_anchors: list of [na, 2] in 'grid' units per level
        """
        na = self.na
        nt = targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        device = self.device

        if nt == 0:
            # placeholder vuoti per tutti i livelli
            for i in range(len(p)):
                tcls.append(torch.zeros(0, dtype=torch.long, device=device))
                tbox.append(torch.zeros(0, 4, device=device))
                indices.append((torch.zeros(0, dtype=torch.long, device=device),) * 4)
                anch.append(torch.zeros(0, 2, device=device))
            return tcls, tbox, indices, anch

        gain = torch.ones(7, device=device)  # (b, c, x, y, w, h, a)
        ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)  # [na, nt]
        # duplichiamo i target per ciascun anchor: [na, nt, 7]
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)

        for i in range(len(p)):
            B, na_l, H, W, _ = p[i].shape
            anchors = scaled_anchors[i]  # [na,2] (aw, ah) in grid

            # mappa target nello spazio di questo livello
            gain[2:6] = torch.tensor([W, H, W, H], device=device)
            t = targets * gain  # [na, nt, 7]

            if t.numel() == 0:
                tcls.append(torch.zeros(0, dtype=torch.long, device=device))
                tbox.append(torch.zeros(0, 4, device=device))
                indices.append((torch.zeros(0, dtype=torch.long, device=device),) * 4)
                anch.append(torch.zeros(0, 2, device=device))
                continue

            # matching anchor per rapporto wh
            r = t[:, :, 4:6] / anchors[:, None]                 # [na, nt, 2]
            j = torch.max(r, 1. / (r + 1e-9)).max(2)[0] < self.hyp['anchor_t']  # [na, nt]
            t = t[j]  # [npos, 7] (se nessun match può essere vuoto)

            if t.shape[0] == 0:
                tcls.append(torch.zeros(0, dtype=torch.long, device=device))
                tbox.append(torch.zeros(0, 4, device=device))
                indices.append((torch.zeros(0, dtype=torch.long, device=device),) * 4)
                anch.append(torch.zeros(0, 2, device=device))
                continue

            # separa campi
            bc, gxy, gwh, a = t[:, :2], t[:, 2:4], t[:, 4:6], t[:, 6].long()
            b = bc[:, 0].long()      # batch index
            c = bc[:, 1].long()      # class
            gij = gxy.long()
            gi, gj = gij[:, 0].clamp_(0, W - 1), gij[:, 1].clamp_(0, H - 1)

            # target box in "offset grid space": dx,dy in cella + wh in grid units
            tbox_i = torch.cat((gxy - gij.float(), gwh), 1)     # [npos,4]
            indices.append((b, a, gj, gi))
            tbox.append(tbox_i)
            anch.append(anchors[a])
            tcls.append(c)

        return tcls, tbox, indices, anch
