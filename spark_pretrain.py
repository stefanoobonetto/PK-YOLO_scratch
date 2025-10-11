# spark_pretrain.py
# SparK-style masked pretraining for RepViT backbone (PK-YOLO)
# - Uses the exact backbone from multimodal_pk_yolo.py
# - Random patch masks; reconstruct masked pixels
# - Save backbone weights for later fine-tuning in PK-YOLO

from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- Your project imports (keep these consistent with your repo) ---
from multimodal_pk_yolo import MultimodalPKYOLO     # must expose .backbone
import config as cfg                                # should contain IMG_SIZE etc.
from brats_dataset import BratsDataset, brats_yolo_collate


# ------------------------------
# Backbone factory (exact same RepViT as detector)
# ------------------------------
def build_repvit_backbone(device: torch.device) -> nn.Module:
    model = MultimodalPKYOLO(cfg)
    backbone = model.backbone
    backbone.to(device)
    backbone.train()
    return backbone


# ------------------------------
# SparK-style Masking
# ------------------------------
def make_patch_mask(x: torch.Tensor, patch: int = 16, mask_ratio: float = 0.6) -> torch.Tensor:
    """
    Create binary mask per image with patch granularity.
    mask=1.0 for visible (unmasked) pixels, 0.0 for masked pixels.
    x: [B, C, H, W]
    """
    B, _, H, W = x.shape
    assert H % patch == 0 and W % patch == 0, "H and W must be divisible by patch size"
    ph, pw = H // patch, W // patch
    total = ph * pw
    num_mask = int(total * mask_ratio)

    device = x.device
    masks = torch.ones(B, total, device=device)
    for i in range(B):
        idx = torch.randperm(total, device=device)[:num_mask]
        masks[i, idx] = 0.0
    masks = masks.view(B, 1, ph, pw)
    masks = F.interpolate(masks, size=(H, W), mode="nearest")  # expand patch mask to pixel mask
    return masks  # [B, 1, H, W] in {0,1}


# ------------------------------
# Lightweight decoder to reconstruct image
# ------------------------------
class ReconstructionDecoder(nn.Module):
    """
    Simple multi-scale decoder. If the backbone returns a list/tuple of feature maps,
    upsample all to input size and fuse; otherwise upsample the single feature map.
    Output channels = 4 (T1, T1ce, T2, FLAIR).
    """
    def __init__(self, out_ch: int = 4, fuse: str = "concat"):
        super().__init__()
        self.fuse = fuse
        # Small convs for channel adaptation before fusion
        self.adapt = nn.ModuleList([nn.Conv2d(256, 128, 1), nn.Conv2d(512, 128, 1), nn.Conv2d(768, 128, 1)])
        # Fallback 1x1 for unknown dims at runtime (will be replaced lazily)
        self.fallback = nn.Conv2d(128, 128, 1)
        # Final head after fusion
        self.head = nn.Sequential(
            nn.Conv2d(128 if fuse == "sum" else 384, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_ch, 1)
        )

    def _adapt_any(self, feat: torch.Tensor) -> torch.Tensor:
        c = feat.shape[1]
        # Choose adapter by common channel sizes, else fallback
        if c == 256:
            return self.adapt[0](feat)
        if c == 512:
            return self.adapt[1](feat)
        if c == 768:
            return self.adapt[2](feat)
        return self.fallback(feat)

    def forward(self, feats, out_hw: tuple[int, int]) -> torch.Tensor:
        H, W = out_hw
        if isinstance(feats, (list, tuple)):
            ups = []
            for f in feats[-3:]:             # use deepest 3 scales if available
                f = self._adapt_any(f)
                f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
                ups.append(f)
            if self.fuse == "sum":
                fused = torch.stack(ups, dim=0).sum(0)
            else:
                fused = torch.cat(ups, dim=1)
        else:
            f = self._adapt_any(feats)
            fused = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
        return self.head(fused)


# ------------------------------
# SparK wrapper (masked input -> backbone -> decoder -> reconstruction)
# ------------------------------
class SparKPretrainModel(nn.Module):
    def __init__(self, backbone: nn.Module, out_ch: int = 4):
        super().__init__()
        self.backbone = backbone
        self.decoder = ReconstructionDecoder(out_ch=out_ch)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, 4, H, W]  (z-scored)
        mask: [B, 1, H, W]  (1=visible, 0=masked)
        """
        x_in = x * mask  # zero-out masked regions (approximate sparse conv behavior)
        feats = self.backbone(x_in)     # list/tuple of feature maps or a single map
        recon = self.decoder(feats, out_hw=(x.shape[-2], x.shape[-1]))  # [B, 4, H, W]
        return recon


# ------------------------------
# Loss: MSE on masked pixels only
# ------------------------------
def masked_mse(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    recon, target: [B, 4, H, W]
    mask:          [B, 1, H, W] where 0=masked (we compute loss on these), 1=visible
    """
    masked_region = 1.0 - mask  # 1 on masked pixels
    diff2 = (recon - target) ** 2
    diff2 = diff2.mean(dim=1, keepdim=True)  # average over channels -> [B,1,H,W]
    num = (diff2 * masked_region).sum()
    den = masked_region.sum() + eps
    return num / den


# ------------------------------
# Training Loop
# ------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # For masked pretraining we want *deterministic, light* transforms:
    # use BratsDataset with training=False (resize + zscore only).
    ds = BratsDataset(items=args.train_items, img_size=cfg.IMG_SIZE, training=False, return_raw=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True, collate_fn=brats_yolo_collate)

    # Build backbone from your detector and wrap with SparK pretrain head
    backbone = build_repvit_backbone(device)
    model = SparKPretrainModel(backbone=backbone, out_ch=4).to(device)

    # Optimizer (paper uses SGD; AdamW also works well for SSL)
    if args.opt.lower() == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        running = 0.0
        for batch in dl:
            imgs = batch["images"].to(device, non_blocking=True)  # [B,4,H,W], already z-scored
            B, C, H, W = imgs.shape

            # Create patch mask and reconstruct masked pixels
            mask = make_patch_mask(imgs, patch=args.patch, mask_ratio=args.mask_ratio).to(imgs.dtype)
            with torch.cuda.amp.autocast(enabled=args.amp):
                recon = model(imgs, mask)              # [B,4,H,W]
                loss = masked_mse(recon, imgs, mask)

            optim.zero_grad(set_to_none=True)
            if args.amp:
                scaler = getattr(train, "_scaler", torch.cuda.amp.GradScaler())
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                train._scaler = scaler
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optim.step()

            running += loss.item()
            global_step += 1

        avg = running / max(1, len(dl))
        print(f"Epoch {epoch+1}/{args.epochs}  |  SparK masked MSE: {avg:.5f}")

        # Save backbone weights (this is what you will load into PK-YOLO)
        ckpt = {
            "epoch": epoch + 1,
            "backbone": model.backbone.state_dict(),
            "img_size": cfg.IMG_SIZE,
            "patch": args.patch,
            "mask_ratio": args.mask_ratio,
        }
        torch.save(ckpt, out_dir / f"repvit_spark_epoch{epoch+1:03d}.pt")

    print("SparK pretraining complete.")


# ------------------------------
# CLI
# ------------------------------
def get_args():
    p = argparse.ArgumentParser("SparK pretraining for PK-YOLO's RepViT backbone")
    p.add_argument("--train-items", type=str, required=True,
                   help="Path to JSON/JSONL/CSV listing image paths (labels ignored).")
    p.add_argument("--out-dir", type=str, default="checkpoints/repvit_spark")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--opt", type=str, default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--patch", type=int, default=16, help="Mask patch size (divides IMG_SIZE).")
    p.add_argument("--mask-ratio", type=float, default=0.6, help="Fraction of patches to mask (0-1).")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)
