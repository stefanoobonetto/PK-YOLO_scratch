import os
import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

MODALITIES = ("t1", "t1ce", "t2", "flair")

def read_gray(path: str, fallback_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Read image as grayscale float32. Returns zeros if missing/corrupt."""
    if path and os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img.astype(np.float32)
    
    # Fallback
    if fallback_hw is not None:
        h, w = fallback_hw
        return np.zeros((h, w), dtype=np.float32)
    return np.zeros((256, 256), dtype=np.float32)


def stack_modalities(paths: Dict[str, str], target_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load and stack 4 modalities --> H x W x 4 (float32)."""
    # Probe shape for fallback
    probe = None
    for m in MODALITIES:
        p = paths.get(m, "")
        if p and os.path.exists(p):
            probe = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if probe is not None:
                break
    
    if probe is not None:
        if probe.ndim == 3:
            probe = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
        hw = probe.shape[:2]
    else:
        hw = target_hw if target_hw is not None else (256, 256)

    imgs = [read_gray(paths.get(m, ""), fallback_hw=hw) for m in MODALITIES]
    return np.stack(imgs, axis=-1).astype(np.float32)


def zscore_per_channel(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score normalization per channel (H W C)."""
    img = img.astype(np.float32)
    for c in range(img.shape[2]):
        v = img[..., c]
        mu = float(v.mean())
        sd = float(v.std())
        if not math.isfinite(mu):
            mu = 0.0
        if not math.isfinite(sd) or sd < eps:
            sd = 1.0
        img[..., c] = (v - mu) / sd
    return img


def add_rician_noise(x: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """Mild Rician noise (MRI-like)."""
    n1 = np.random.normal(0.0, sigma, x.shape).astype(np.float32)
    n2 = np.random.normal(0.0, sigma, x.shape).astype(np.float32)
    return np.sqrt((x + n1) ** 2 + n2 ** 2)


def apply_bias_field(x: np.ndarray, max_scale: float = 0.12) -> np.ndarray:
    """Low-frequency multiplicative bias field."""
    h, w, _ = x.shape
    low = 32
    field = cv2.resize(np.random.randn(low, low).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    sigma = 0.15 * max(h, w)
    field = cv2.GaussianBlur(field, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    field -= field.min()
    denom = field.max() - field.min() + 1e-6
    field = 2.0 * (field / denom) - 1.0
    scale = 1.0 + max_scale * field
    return x * scale[..., None]


def _mild_intensity_ops(x: np.ndarray) -> np.ndarray:
    """Small brightness/contrast and gamma adjustments."""
    # Brightness/contrast
    if np.random.rand() < 0.35:
        alpha = 1.0 + np.random.uniform(-0.1, 0.1)
        beta = np.random.uniform(-0.1, 0.1)
        x = x * alpha + beta
    
    # Gamma
    if np.random.rand() < 0.25:
        y = x.copy()
        for c in range(y.shape[2]):
            v = y[..., c]
            p1, p99 = np.percentile(v, [1, 99])
            v = np.clip((v - p1) / (p99 - p1 + 1e-6), 0, 1)
            gamma = np.random.uniform(0.9, 1.1)
            v = v ** gamma
            v = (v - 0.5) * 2.0
            y[..., c] = v
        x = y
    return x


def domain_noise_ops(x: np.ndarray) -> np.ndarray:
    """Light Rician noise and Gaussian blur."""
    if np.random.rand() < 0.20:
        x = add_rician_noise(x, sigma=0.02)
    if np.random.rand() < 0.20:
        x = np.stack(
            [cv2.GaussianBlur(x[..., c], (3, 3), sigmaX=0.5).astype(np.float32) for c in range(x.shape[2])],
            axis=-1
        )
    return x


def build_transforms(img_size: int, training: bool) -> A.Compose:
    """Albumentations pipeline for MRI. BBoxes in YOLO format (cx, cy, w, h) normalized."""
    bbox_params = A.BboxParams(
        format="yolo",
        min_visibility=0.40,
        label_fields=["class_labels"],
        check_each_transform=False,
    )

    if training:
        geom = [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.10, rotate_limit=10,
                                   border_mode=cv2.BORDER_REFLECT_101, p=1.0),
                A.Affine(scale=(0.95, 1.05), translate_percent=(0.02, 0.02),
                         rotate=(-10, 10), shear=(-5, 5), cval=0,
                         mode=cv2.BORDER_REFLECT_101, p=1.0),
            ], p=0.60),
            A.HorizontalFlip(p=0.50),
            A.VerticalFlip(p=0.20),
            A.ElasticTransform(alpha=8, sigma=8 * 0.07, alpha_affine=2,
                               border_mode=cv2.BORDER_REFLECT_101, p=0.10),
        ]
        intensity = [
            A.Lambda(image=zscore_per_channel),
            A.Lambda(image=lambda x, **k: apply_bias_field(x, 0.12) if np.random.rand() < 0.25 else x),
            A.Lambda(image=lambda x, **k: _mild_intensity_ops(x)),
            A.Lambda(image=lambda x, **k: domain_noise_ops(x)),
        ]
        return A.Compose(geom + intensity + [ToTensorV2()], bbox_params=bbox_params)

    # Validation
    return A.Compose(
        [A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
         A.Lambda(image=zscore_per_channel),
         ToTensorV2()],
        bbox_params=bbox_params
    )


def _load_items_from_file(path: str) -> List[Dict[str, Any]]:
    """
    Load items from JSON/JSONL/CSV.
    Expected fields: {t1, t1ce, t2, flair, bboxes, labels, id}
    """
    ext = os.path.splitext(path)[1].lower()
    items: List[Dict[str, Any]] = []

    if ext in (".json", ".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            if ext == ".jsonl":
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
            else:
                obj = json.load(f)
                if isinstance(obj, dict) and "items" in obj:
                    items = obj["items"]
                elif isinstance(obj, list):
                    items = obj
                else:
                    raise ValueError("Unsupported JSON structure.")
        return items

    if ext == ".csv":
        import csv
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = {
                    "t1": row.get("t1", ""),
                    "t1ce": row.get("t1ce", ""),
                    "t2": row.get("t2", ""),
                    "flair": row.get("flair", ""),
                    "bboxes": json.loads(row.get("bboxes", "[]")),
                    "labels": json.loads(row.get("labels", "[]")),
                    "id": row.get("id", None),
                }
                items.append(item)
        return items

    raise ValueError(f"Unsupported file type: {path}")


class BratsDataset(Dataset):
    """
    Multimodal MRI dataset for detection.
    
    Args:
        items: Path to JSON/JSONL/CSV or pre-built list of dicts
        img_size: Output size (square)
        training: Whether to use training augmentations
        return_raw: If True, also return non-augmented image
    """

    def __init__(
        self,
        items: Union[str, List[Dict[str, Any]]],
        img_size: int = 640,
        training: bool = True,
        return_raw: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(items, str):
            self.items = _load_items_from_file(items)
        else:
            self.items = items
        
        if len(self.items) == 0:
            raise ValueError("No dataset items found.")

        self.img_size = int(img_size)
        self.training = bool(training)
        self.return_raw = bool(return_raw)

        self.transform = build_transforms(self.img_size, training=self.training)
        self.raw_transform = A.Compose(
            [A.Resize(self.img_size, self.img_size), A.Lambda(image=zscore_per_channel)]
        )

    def __len__(self) -> int:
        return len(self.items)

    def _extract_paths(self, item: Dict[str, Any]) -> Dict[str, str]:
        return {
            "t1": item.get("t1", ""),
            "t1ce": item.get("t1ce", ""),
            "t2": item.get("t2", ""),
            "flair": item.get("flair", ""),
        }

    def _extract_ann(self, item: Dict[str, Any]) -> Tuple[List[List[float]], List[int]]:
        bboxes = item.get("bboxes", [])
        labels = item.get("labels", [])
        if not isinstance(bboxes, list):
            raise ValueError("bboxes must be list of [cx, cy, w, h] (normalized).")
        if not isinstance(labels, list):
            raise ValueError("labels must be list of ints.")
        return bboxes, labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        id_str = item.get("id", None)

        paths = self._extract_paths(item)
        image = stack_modalities(paths)

        if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        bboxes, labels = self._extract_ann(item)

        out = self.transform(image=image, bboxes=bboxes, class_labels=labels)

        sample: Dict[str, Any] = {
            "image": out["image"],
            "bboxes": np.array(out["bboxes"], dtype=np.float32),
            "labels": np.array(out["class_labels"], dtype=np.int64),
            "id": id_str,
        }

        if self.return_raw:
            rr = self.raw_transform(image=image)["image"]
            sample["raw_image"] = rr

        return sample


def brats_yolo_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for detection training.
    Returns dict with: images, bboxes, labels, ids, raw_images (optional)
    """
    images = torch.stack([b["image"] for b in batch], dim=0).float()

    bboxes_list = []
    labels_list = []
    ids = [b.get("id", None) for b in batch]

    for b in batch:
        bb = torch.from_numpy(b["bboxes"]).float() if isinstance(b["bboxes"], np.ndarray) else \
             torch.as_tensor(b["bboxes"], dtype=torch.float32)
        ll = torch.from_numpy(b["labels"]).long() if isinstance(b["labels"], np.ndarray) else \
             torch.as_tensor(b["labels"], dtype=torch.long)
        bboxes_list.append(bb)
        labels_list.append(ll)

    out = {
        "images": images,
        "bboxes": bboxes_list,
        "labels": labels_list,
        "ids": ids,
    }

    if "raw_image" in batch[0]:
        raws = [torch.from_numpy(b["raw_image"]).permute(2, 0, 1).float() for b in batch]
        out["raw_images"] = torch.stack(raws, dim=0)

    return out

def build_weighted_sampler_for_small_objects(
    items: List[Dict[str, Any]],
    small_thr: float = 0.002,
    small_boost: float = 3.0,
) -> torch.utils.data.WeightedRandomSampler:
    """Oversample slices containing small boxes (area < small_thr)."""
    weights = []
    for it in items:
        bboxes = it.get("bboxes", [])
        has_small = any((b[2] * b[3]) < small_thr for b in bboxes)
        weights.append(small_boost if has_small else 1.0)

    total = len(weights)
    return torch.utils.data.WeightedRandomSampler(weights, num_samples=total, replacement=True)