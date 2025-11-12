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
from tqdm.auto import tqdm
from contextlib import nullcontext

# --- Your project imports (keep these consistent with your repo) ---
from multimodal_pk_yolo import MultimodalPKYOLO     # must expose .backbone
from utils.config import SimpleConfig, create_default_config
from brats_dataset import BraTSDataset, collate_fn

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ------------------------------
# Backbone factory (exact same RepViT as detector)
# ------------------------------
def build_repvit_backbone(device: torch.device, cfg) -> nn.Module:
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
    masks = F.interpolate(masks, size=(H, W), mode="nearest")
    return masks  # [B,1,H,W] in {0,1}


# ------------------------------
# Lightweight decoder to reconstruct image
# ------------------------------
class ReconstructionDecoder(nn.Module):
    """
    Multi-scale decoder. If the backbone returns a list/tuple of feature maps,
    upsample to input size and fuse; otherwise upsample the single feature map.
    Output channels = 4 (T1, T1ce, T2, FLAIR).
    """
    def __init__(self, out_ch: int = 4, fuse: str = "concat"):
        super().__init__()
        self.fuse = fuse
        self.adapt = nn.ModuleList([
            nn.Conv2d(256, 128, 1),
            nn.Conv2d(512, 128, 1),
            nn.Conv2d(768, 128, 1)
        ])
        self.fallback = nn.Conv2d(128, 128, 1)
        self.head = nn.Sequential(
            nn.Conv2d(128 if fuse == "sum" else 384, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_ch, 1)
        )

    def _adapt_any(self, feat: torch.Tensor) -> torch.Tensor:
        c = feat.shape[1]
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
            for f in feats[-3:]:
                f = self._adapt_any(f)
                f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
                ups.append(f)
            fused = torch.stack(ups, dim=0).sum(0) if self.fuse == "sum" else torch.cat(ups, dim=1)
        else:
            f = self._adapt_any(feats)
            fused = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
        return self.head(fused)


# ------------------------------
# SparK wrapper
# ------------------------------
class SparKPretrainModel(nn.Module):
    def __init__(self, backbone: nn.Module, out_ch: int = 4):
        super().__init__()
        self.backbone = backbone
        self.decoder = ReconstructionDecoder(out_ch=out_ch)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_in = x * mask
        feats = self.backbone(x_in)
        recon = self.decoder(feats, out_hw=(x.shape[-2], x.shape[-1]))
        return recon


# ------------------------------
# Loss: MSE on masked pixels only
# ------------------------------
def masked_mse(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    masked_region = 1.0 - mask  # 1 on masked pixels
    diff2 = (recon - target) ** 2
    diff2 = diff2.mean(dim=1, keepdim=True)  # [B,1,H,W]
    num = (diff2 * masked_region).sum()
    den = masked_region.sum() + eps
    return num / den


# ------------------------------
# Evaluation (no grad)
# ------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, device: torch.device, args) -> float:
    model.eval()
    total = 0.0
    iters = 0
    amp_ctx = torch.amp.autocast('cuda', enabled=args.amp) if torch.cuda.is_available() else nullcontext()
    for batch in dl:
        imgs = batch["images"].to(device, non_blocking=True)
        mask = make_patch_mask(imgs, patch=args.patch, mask_ratio=args.mask_ratio).to(imgs.dtype)
        with amp_ctx:
            recon = model(imgs, mask)
            loss = masked_mse(recon, imgs, mask)
        total += float(loss.item())
        iters += 1
    return total / max(1, iters)


# ------------------------------
# Training Loop (with Early Stopping on val loss)
# ------------------------------
def train(args, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    img_size = cfg.get('model.img_size')
    assert img_size % args.patch == 0, f"IMG_SIZE ({img_size}) must be divisible by patch ({args.patch})."

    data_dir = Path(args.data_dir)

    # --- Train loader
    ds = BraTSDataset(
        data_dir=data_dir,  
        split='train', 
        img_size=img_size, 
        augment=False
    )
    
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
        collate_fn=collate_fn
    )

    # --- Val loader (data_dir / 'val')
    dl_val = None
    # val_root = data_dir / 'val'
    val_root = data_dir / 'val'
    if val_root.exists():
        ds_val = BraTSDataset(
            data_dir=data_dir, 
            split='val', 
            img_size=img_size, 
            augment=False
        )
        
        dl_val = DataLoader(
            ds_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=False,
            collate_fn=collate_fn
        )
    else:
        print(f"[WARN] Validation directory not found: {val_root}. Early stopping will be disabled.")

    backbone = build_repvit_backbone(device, cfg)
    model = SparKPretrainModel(backbone=backbone, out_ch=4).to(device)

    optim = (torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
             if args.opt.lower() == "sgd" else
             torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    best_metric = float("inf")
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        pbar = tqdm(dl, total=len(dl), desc=f"Epoch {epoch+1}/{args.epochs}", leave=False, mininterval=1.0)
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

        train_avg = running / max(1, len(dl))

        # --- Validation & early stopping metric
        if dl_val is not None:
            val_loss = evaluate(model, dl_val, device, args)
            print(f"Epoch {epoch+1}/{args.epochs} | train {train_avg:.5f} | val {val_loss:.5f}")
            metric = val_loss
        else:
            print(f"Epoch {epoch+1}/{args.epochs} | train {train_avg:.5f}")
            metric = train_avg  # fallback (no early stop)

        # --- Save checkpoint of this epoch
        ckpt_payload = {
            "epoch": epoch + 1,
            "backbone": model.backbone.state_dict(),
            "backbone_state_dict": model.backbone.state_dict(),  # compatibility
            "img_size": img_size,
            "patch": args.patch,
            "mask_ratio": args.mask_ratio,
            "train_loss": train_avg,
        }
        if dl_val is not None:
            ckpt_payload["val_loss"] = metric

        ckpt_path = out_dir / f"repvit_spark_epoch{epoch+1:03d}.pt"
        torch.save(ckpt_payload, ckpt_path)

        # --- Best model logic
        improved = (metric + args.min_delta) < best_metric
        if improved:
            best_metric = metric
            epochs_no_improve = 0
            best_path = out_dir / "best_spark_model.pth"
            torch.save(ckpt_payload, best_path)
            print(f"  ↳ new best: {best_metric:.5f}  (saved {best_path})")
        else:
            epochs_no_improve += 1

        # --- Early stopping (only if we have validation)
        if dl_val is not None and args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch+1}: no improvement ≥ {args.min_delta} "
                  f"for {args.patience} consecutive epochs on validation loss.")
            break

    print("SparK pretraining complete.")


# ------------------------------
# CLI
# ------------------------------
def get_args():
    p = argparse.ArgumentParser("SparK pretraining for PK-YOLO's RepViT backbone")
    p.add_argument("--data-dir", type=str, required=True, help="Root dataset directory containing train/ and val/ subfolders")
    p.add_argument("--split", type=str, default="train")  # kept for compatibility; train split is enforced anyway
    p.add_argument("--out-dir", type=str, default="checkpoints/repvit_spark")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--opt", type=str, default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--patch", type=int, default=16, help="Mask patch size (divides IMG_SIZE).")
    p.add_argument("--mask-ratio", type=float, default=0.75, help="Fraction of patches to mask (0-1).")
    p.add_argument("--img-size", type=int, default=224, help="Pretraining image size (e.g., 224 or 256).")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision training")

    # Early stopping controls
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience on val loss (epochs)")
    p.add_argument("--min-delta", type=float, default=1e-4, help="Minimum improvement to reset patience")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    cfg = SimpleConfig(create_default_config(args))

    # Ensure the image size used downstream matches the CLI
    try:
        cfg.set('model.img_size', int(args.img_size))
    except AttributeError:
        try:
            cfg['model.img_size'] = int(args.img_size)
        except Exception:
            pass

    train(args, cfg)
