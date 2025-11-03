import math
import torch
import torch.nn as nn

EPS = 1e-7


class YOLOLoss(nn.Module):
    """
    Robust YOLO loss for small-tumor detection.
    - Objectness only (no class term).
    - Focal modulation on objectness.
    - Safe target building & IoU math (no NaNs/Inf).
    """

    def __init__(self, model, anchors=None, img_size=640):
        super().__init__()
        self.device = next(model.parameters()).device
        self.img_size = int(img_size)

        # Loss weights & knobs (tuned for small objects)
        self.hyp = {
            "box": 0.10,        # box regression loss weight
            "obj": 1.00,        # objectness loss weight
            "obj_pw": 1.50,     # positive weight for obj BCE
            "anchor_t": 3.00,   # anchor match threshold (ratio)
            "fl_gamma": 1.50,   # focal gamma for obj
        }

        # Anchors: prefer model buffer
        if anchors is not None:
            self.anchors = anchors.to(self.device)
        elif hasattr(model, "anchors"):
            self.anchors = model.anchors.detach().to(self.device)
        else:
            raise ValueError("Anchors not provided and model has no 'anchors' buffer.")
        self.nl = int(self.anchors.shape[0])   # number of layers
        self.na = int(self.anchors.shape[1])   # anchors per layer

        # Per-layer balance (emphasize high-resolution heads)
        self.balance = [8.0, 4.0, 1.0, 0.4][: self.nl] if self.nl >= 3 else [4.0, 1.0][: self.nl]

        # BCE with pos_weight; reduction='none' so we can focal-modulate
        self.register_buffer("_pos_weight", torch.tensor(self.hyp["obj_pw"], device=self.device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight, reduction="none")

    # --------------------------- Public API --------------------------- #

    def forward(self, predictions, targets):
        """
        Args:
            predictions: list of length nl; each item is (bbox_pred, obj_pred)
                         bbox_pred: [B, na*4, H, W], obj_pred: [B, na*1, H, W]
            targets: dict with
                     - 'images': [B, C, H, W]
                     - 'bboxes': [B, M, 4] in cx,cy,w,h (normalized [0,1] or pixels)
                     - 'labels': [B, M] (unused here)
        Returns:
            loss: scalar tensor
            components: tensor([lbox, lobj]) detached
        """
        device = self.device

        # Targets -> [nt, 5] with columns [batch_idx, cx, cy, w, h] normalized in [0,1]
        targets_tensor = self._prepare_targets(targets)

        # Predictions -> list of [B, na, H, W, 5]
        p = self._prepare_predictions(predictions)

        # Build targets for each layer (safe in all edge cases)
        tbox, indices, anch = self._build_targets(p, targets_tensor)

        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)

        for i, pi in enumerate(p):
            # pi: [B, na, H, W, 5], last dim = (tx, ty, tw, th, tobj_logit)
            b, a, gj, gi = indices[i]  # each: [n_pos]
            tobj = torch.zeros_like(pi[..., 4])  # [B, na, H, W]

            n_pos = b.numel()
            if n_pos > 0:
                ps = pi[b, a, gj, gi]  # [n_pos, 5]

                # Decode (cell-offset param) to grid-space center/size
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5       # [-0.5, 1.5]
                pwh = (ps[:, 2:4].sigmoid() * 2.0) ** 2.0   # positive
                pwh = pwh * anch[i]                          # anchor-scaled (grid units)
                pbox = torch.cat((pxy, pwh), 1)              # [n_pos, 4]

                # Target boxes already in grid units & cell-offset form:
                # tbox[i] = [gxy - floor(gxy), gwh]
                iou = self._bbox_iou(pbox, tbox[i], xywh=True, CIoU=True).clamp(0.0, 1.0)

                # Focal-like box loss on (1 - IoU)
                lbox += ((1.0 - iou) ** self.hyp["fl_gamma"]).mean() * self.balance[i]

                # Objectness soft target is IoU
                tobj[b, a, gj, gi] = iou.detach().type_as(tobj)

            # BCE objectness with focal modulation
            obj_logit = pi[..., 4]
            obj_loss = self.BCEobj(obj_logit, tobj)

            # Focal modulation (no extra clamp needed; BCE is stable)
            p_obj = obj_logit.sigmoid()
            gamma = self.hyp["fl_gamma"]
            focal = tobj * (1.0 - p_obj).pow(gamma) + (1.0 - tobj) * p_obj.pow(gamma)

            lobj += (obj_loss * focal).mean() * self.balance[i]

        # Weighted sum
        lbox = lbox * self.hyp["box"]
        lobj = lobj * self.hyp["obj"]

        # Final loss (safe)
        loss = (lbox + lobj)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)

        comp = torch.stack([
            torch.nan_to_num(lbox.detach(), nan=0.0, posinf=1e4, neginf=0.0),
            torch.nan_to_num(lobj.detach(), nan=0.0, posinf=1e4, neginf=0.0),
        ]).squeeze()

        return loss, comp

    # --------------------------- Helpers --------------------------- #

    def _prepare_targets(self, targets):
        """
        Dict -> [nt, 5] (b, cx, cy, w, h) all normalized to [0,1].
        Drops degenerate zeros. Returns empty tensor if no GT.
        """
        if not isinstance(targets, dict):
            return targets

        device = self.device
        B = targets["images"].shape[0]
        T = []

        for i in range(B):
            bboxes = targets["bboxes"][i]  # [M,4]
            if bboxes.numel() == 0:
                continue

            # Keep only non-zero boxes
            valid = (bboxes[..., 2] > 0) & (bboxes[..., 3] > 0)
            if valid.any():
                boxes = bboxes[valid].clone()

                # Normalize if they look like pixels
                if boxes.max() > 1.01:
                    s = float(self.img_size)
                    boxes = boxes / s

                # Clamp to [0,1]
                boxes = boxes.clamp_(0.0, 1.0)

                # batch index column
                bi = torch.full((boxes.shape[0], 1), i, device=device, dtype=boxes.dtype)
                T.append(torch.cat([bi, boxes.to(device)], dim=1))

        if len(T) == 0:
            return torch.zeros((0, 5), device=device)

        return torch.cat(T, dim=0)

    def _prepare_predictions(self, predictions):
        """
        (bbox_pred, obj_pred) -> [B, na, H, W, 5] per level.
        bbox_pred: [B, na*4, H, W]
        obj_pred : [B, na*1, H, W]
        """
        out = []
        for bbox_pred, obj_pred in predictions:
            B, _, H, W = bbox_pred.shape
            # reshape
            bbox = bbox_pred.view(B, self.na, 4, H, W).permute(0, 1, 3, 4, 2).contiguous()
            obj  = obj_pred.view(B, self.na, 1, H, W).permute(0, 1, 3, 4, 2).contiguous()
            out.append(torch.cat([bbox, obj], dim=-1))  # [B, na, H, W, 5]
        return out

    def _build_targets(self, p, targets):
        """
        Match targets to anchors per layer.
        Returns per layer:
          tbox[i]:    [n_pos, 4] (tx, ty, tw, th) in grid units
          indices[i]: tuple(b, a, gj, gi) each [n_pos] long
          anch[i]:    [n_pos, 2] anchor widths/heights (grid units)
        """
        device = self.device
        nl = len(p)
        tbox, indices, anch = [], [], []

        nt = targets.shape[0]
        if nt == 0:
            # return properly shaped empties
            for _ in range(nl):
                tbox.append(torch.zeros(0, 4, device=device))
                indices.append((
                    torch.zeros(0, dtype=torch.long, device=device),  # b
                    torch.zeros(0, dtype=torch.long, device=device),  # a
                    torch.zeros(0, dtype=torch.long, device=device),  # gj
                    torch.zeros(0, dtype=torch.long, device=device),  # gi
                ))
                anch.append(torch.zeros(0, 2, device=device))
            return tbox, indices, anch

        # Strides from feature map size
        strides = []
        for pi in p:
            _, _, H, W, _ = pi.shape
            strides.append(self.img_size / float(H))  # assume square

        # Anchors scaled to grid space per layer
        scaled_anchors = [self.anchors[i] / strides[i] for i in range(nl)]

        # Repeat targets for anchors & append anchor index
        # targets: [nt, 5] -> [na, nt, 6] after cat (b, cx, cy, w, h, a)
        ai = torch.arange(self.na, device=targets.device).view(self.na, 1)
        ai = ai.expand(self.na, nt) if nt > 0 else ai[:, :0]  # [na, nt] or [na, 0]
        t = torch.cat((targets.repeat(self.na, 1, 1), ai.unsqueeze(-1)), dim=2)

        # Gain to map normalized [cx,cy,w,h] -> grid coords per layer
        gain = torch.ones(6, device=targets.device)  # (b, cx, cy, w, h, a)

        for i in range(nl):
            pi = p[i]
            anchors_i = scaled_anchors[i].to(device)  # [na,2] in grid units
            _, _, H, W, _ = pi.shape

            # set gain for this layer
            gain[1:5] = torch.tensor((W, H, W, H), device=targets.device, dtype=gain.dtype)

            # scale to grid
            t_layer = (t * gain).clone()  # [na, nt, 6]

            # Anchor match
            # r = (gw, gh) / (aw, ah), keep if both ratios < anchor_t (on max dimension)
            r = t_layer[:, :, 3:5] / (anchors_i[:, None, :] + EPS)  # [na, nt, 2]
            r_max = torch.max(r, 1.0 / (r + EPS)).amax(dim=2)       # [na, nt]
            mask = r_max < self.hyp["anchor_t"]

            # Flatten matches
            if mask.any():
                t_sel = t_layer[mask]  # [n_pos, 6]
                b = t_sel[:, 0].long()
                gxy = t_sel[:, 1:3]           # grid coords (float)
                gwh = t_sel[:, 3:5]           # grid sizes (float)
                a = t_sel[:, 5].long()        # anchor id

                gi = gxy[:, 0].long().clamp_(0, W - 1)
                gj = gxy[:, 1].long().clamp_(0, H - 1)

                # Offsets inside the cell (tx, ty) = gxy - floor(gxy)
                txy = (gxy - torch.stack((gi, gj), dim=1).float()).clamp_(0.0, 1.0)

                indices.append((b, a, gj, gi))
                tbox.append(torch.cat((txy, gwh), dim=1))      # [n_pos, 4]
                anch.append(anchors_i[a])                      # [n_pos, 2]
            else:
                indices.append((
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                ))
                tbox.append(torch.zeros(0, 4, device=device))
                anch.append(torch.zeros(0, 2, device=device))

        return tbox, indices, anch

    @staticmethod
    def _bbox_iou(box1, box2, xywh=True, CIoU=False):
        """
        IoU / CIoU between two sets of boxes.
        box1, box2: [N,4]
        If xywh=True, boxes are (cx, cy, w, h) in same units.
        """
        eps = EPS

        if xywh:
            # cxcywh -> xyxy
            b1_x1 = box1[:, 0] - box1[:, 2] * 0.5
            b1_y1 = box1[:, 1] - box1[:, 3] * 0.5
            b1_x2 = box1[:, 0] + box1[:, 2] * 0.5
            b1_y2 = box1[:, 1] + box1[:, 3] * 0.5

            b2_x1 = box2[:, 0] - box2[:, 2] * 0.5
            b2_y1 = box2[:, 1] - box2[:, 3] * 0.5
            b2_x2 = box2[:, 0] + box2[:, 2] * 0.5
            b2_y2 = box2[:, 1] + box2[:, 3] * 0.5
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim=1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim=1)

        # Intersection
        inter_w = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(0.0)
        inter_h = (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clamp(0.0)
        inter = inter_w * inter_h

        # Areas & union
        w1 = (b1_x2 - b1_x1).clamp_min(0.0)
        h1 = (b1_y2 - b1_y1).clamp_min(0.0)
        w2 = (b2_x2 - b2_x1).clamp_min(0.0)
        h2 = (b2_y2 - b2_y1).clamp_min(0.0)
        union = (w1 * h1) + (w2 * h2) - inter
        iou = inter / (union + eps)

        if not CIoU:
            return iou

        # CIoU terms
        cw = torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)
        ch = torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)
        c2 = (cw * cw + ch * ch).clamp_min(eps)

        # center distance
        b1_cx = (b1_x1 + b1_x2) * 0.5
        b1_cy = (b1_y1 + b1_y2) * 0.5
        b2_cx = (b2_x1 + b2_x2) * 0.5
        b2_cy = (b2_y1 + b2_y2) * 0.5
        rho2 = (b2_cx - b1_cx) ** 2 + (b2_cy - b1_cy) ** 2

        v = (4.0 / (math.pi ** 2)) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))).pow(2)
        with torch.no_grad():
            alpha = v / (1.0 - iou + v + eps)

        return iou - (rho2 / c2 + v * alpha)
