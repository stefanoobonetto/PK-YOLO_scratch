# spark_pretrain.py
# SparK-style masked pretraining for RepViT backbone (PK-YOLO)
# - Uses the exact backbone from multimodal_pk_yolo.py
# - Random patch masks; reconstruct masked pixels
# - Save backbone weights for later fine-tuning in PK-YOLO

from __future__ import annotations
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- Your project imports (keep these consistent with your repo) ---
from multimodal_pk_yolo import MultimodalPKYOLO     # must expose .backbone
from utils.config import SimpleConfig, create_default_config                                # should contain IMG_SIZE etc.
from brats_dataset import BraTSDataset, collate_fn


# ------------------------------
# Backbone factory (exact same RepViT as detector)
# ------------------------------
# spark_pretrain.py
def build_repvit_backbone(device: torch.device, cfg) -> nn.Module:
    # Try common constructor signatures explicitly to avoid misbinding
    model = None
    last_err = None
    for attempt in (
        dict(cfg=cfg, input_channels=4),
        dict(input_channels=4, cfg=cfg),
        dict(input_channels=4),
        dict(cfg=cfg),
    ):
        try:
            model = MultimodalPKYOLO(**attempt)
            break
        except TypeError as e:
            last_err = e
    if model is None:
        raise TypeError(
            "Could not construct MultimodalPKYOLO with any common signature. "
            "Please check its __init__ and ensure it accepts cfg and/or input_channels."
        ) from last_err

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
def train(args, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    img_size = cfg.get('model.img_size')
    assert img_size % args.patch == 0, f"IMG_SIZE ({img_size}) must be divisible by patch ({args.patch})."

    data_dir = Path(args.data_dir)
    split = args.split

    ds = BraTSDataset(
        data_dir=data_dir,
        split=split,
        img_size=img_size,
        augment=False  # pretrain: keep stable inputs
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    backbone = build_repvit_backbone(device, cfg)
    model = SparKPretrainModel(backbone=backbone, out_ch=4).to(device)

    optim = (torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
             if args.opt.lower() == "sgd" else
             torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from tqdm.auto import tqdm
    from contextlib import nullcontext
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        pbar = tqdm(dl, total=len(dl), desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch in pbar:
            imgs = batch["images"].to(device, non_blocking=True)
            mask = make_patch_mask(imgs, patch=args.patch, mask_ratio=args.mask_ratio).to(imgs.dtype)

            amp_ctx = torch.amp.autocast('cuda', enabled=args.amp) if torch.cuda.is_available() else nullcontext()
            with amp_ctx:
                recon = model(imgs, mask)
                loss = masked_mse(recon, imgs, mask)

            optim.zero_grad(set_to_none=True)
            if args.amp:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optim); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optim.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.5f}", avg=f"{running / max(1, pbar.n):.5f}")

        avg = running / max(1, len(dl))
        print(f"Epoch {epoch+1}/{args.epochs} | SparK masked MSE: {avg:.5f}")

        ckpt_path = out_dir / f"repvit_spark_epoch{epoch+1:03d}.pt"
        torch.save({
            "epoch": epoch + 1,
            "backbone": model.backbone.state_dict(),
            "img_size": img_size,
            "patch": args.patch,
            "mask_ratio": args.mask_ratio,
            "avg_loss": avg,
        }, ckpt_path)

        # keep a copy named best_spark_model.pth
        if avg < best_loss:
            best_loss = avg
            best_path = out_dir / "best_spark_model.pth"
            torch.save({
                "epoch": epoch + 1,
                "backbone": model.backbone.state_dict(),
                "img_size": img_size,
                "patch": args.patch,
                "mask_ratio": args.mask_ratio,
                "avg_loss": avg,
            }, best_path)
            print(f"  ↳ new best: {avg:.5f}  (saved {best_path})")

    print("SparK pretraining complete.")


# ------------------------------
# CLI
# ------------------------------
# --- replace get_args() entirely ---
def get_args():
    p = argparse.ArgumentParser("SparK pretraining for PK-YOLO's RepViT backbone")
    p.add_argument("--data-dir", type=str, required=True,
                   help="Root of dataset containing split/images and split/labels.")
    p.add_argument("--split", type=str, default="train", help="Dataset split (train/val)")

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
    cfg = SimpleConfig(create_default_config(args))
    train(args, cfg)
