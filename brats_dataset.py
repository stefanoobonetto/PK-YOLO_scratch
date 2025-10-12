import logging
from pathlib import Path
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------------------------------------------------
# OpenCV thread safety (sometimes speeds up DataLoader on Windows)
# ---------------------------------------------------------------------
try:
    import cv2 as _cv2
    _cv2.setNumThreads(0)
except Exception:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Top-level, picklable z-score function (no lambda; accepts **kwargs)
# ---------------------------------------------------------------------
def zscore_4ch(img: np.ndarray, **kwargs) -> np.ndarray:
    """
    Per-image, per-channel z-score for a 4-channel MRI slice.
    Albumentations may pass extra kwargs; accept **kwargs.
    Expects HxWxC (C=4 preferred). Returns float32.
    """
    x = img.astype(np.float32)
    if x.ndim == 2:
        # Upgrade to HxWx1 if somehow grayscale slipped through
        x = x[..., None]

    # Be robust to channels != 4 (pad/truncate). Prefer to keep 4 channels.
    c = x.shape[-1]
    if c < 4:
        pads = 4 - c
        x = np.concatenate([x, np.zeros((*x.shape[:2], pads), dtype=x.dtype)], axis=-1)
    elif c > 4:
        x = x[..., :4]

    mean = x.mean(axis=(0, 1), keepdims=True)
    std = x.std(axis=(0, 1), keepdims=True)
    x = (x - mean) / (std + 1e-6)
    return x.astype(np.float32)

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class BraTSDataset(Dataset):
    """
    BraTS-style 2D slice dataset with 4 MRI modalities stacked as channels:
    [t1, t1ce, t2, flair] -> image: HxWx4 (float32 after transforms).

    Directory layout:
      data_dir/
        <split>/images/*.png
        <split>/labels/*.txt  (YOLO format per slice_id)

    Each slice_id is like: BraTS20_Training_###_slice_### (or an alt form).
    """

    def __init__(self, data_dir, split='train', img_size=640, augment=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = int(img_size)
        self.augment = bool(augment) and split == 'train'

        self.image_dir = self.data_dir / split / 'images'
        self.label_dir = self.data_dir / split / 'labels'

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.label_dir}")

        self.slice_ids = self._get_slice_ids()
        self._setup_transforms()

        logger.info(f"Loaded {len(self.slice_ids)} samples for split='{split}'")

    # ------------------------------
    def _get_slice_ids(self):
        # Extract unique slice IDs from filenames
        slice_ids = set()
        image_files = list(self.image_dir.glob('*.png')) + list(self.image_dir.glob('*.PNG'))

        # Main pattern: BraTS20_Training_###_slice_###
        pat_main = re.compile(r'(BraTS20_Training_\d+_slice_\d+)_([tf]\w*|flair)\.png', re.IGNORECASE)
        # Alt pattern: BraTS20_Training_###_###
        pat_alt = re.compile(r'(BraTS20_Training_\d+_\d+)_([tf]\w*|flair)\.png', re.IGNORECASE)

        for img_file in image_files:
            m = pat_main.match(img_file.name)
            if m:
                slice_ids.add(m.group(1))
                continue
            m2 = pat_alt.match(img_file.name)
            if m2:
                slice_ids.add(m2.group(1))

        return sorted(slice_ids)

    # ------------------------------
    def _setup_transforms(self):
        common_bbox = A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.1
        )

        if self.augment:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),

                # Gentle, bbox-safe spatial jitter (MRI-friendly)
                A.Affine(
                    scale=(0.9, 1.1),                 # ±10%
                    translate_percent=(0.0, 0.04),    # up to 4%
                    rotate=(-8, 8),                   # small rotations
                    shear=0,                          # explicit 0 (not None)
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_CONSTANT,  # use border_mode (no 'mode'/'cval')
                    fit_output=False,
                    p=0.6
                ),

                A.HorizontalFlip(p=0.5),

                # Scanner-like intensity shifts (mild)
                A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.4),
                A.RandomGamma(gamma_limit=(90, 110), p=0.3),

                # Light artifacts (kept mild; GaussNoise removed for API compat)
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, p=0.25),
                A.GaussianBlur(blur_limit=(3, 5), p=0.15),

                # Per-image z-score (picklable)
                A.Lambda(image=zscore_4ch),

                ToTensorV2()
            ], bbox_params=common_bbox)
        else:
            # Eval/Pretrain: stable, no heavy augs; still z-score per image
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Lambda(image=zscore_4ch),
                ToTensorV2()
            ], bbox_params=common_bbox)

    # ------------------------------
    def _read_modality(self, slice_id: str, modality: str) -> np.ndarray:
        """Read a single modality, grayscale, resized to img_size if needed."""
        for ext in ("png", "PNG"):
            path = self.image_dir / f"{slice_id}_{modality}.{ext}"
            if path.exists():
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    break
                if img.shape[:2] != (self.img_size, self.img_size):
                    img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
                return img
        # Fallback blank if missing
        return np.zeros((self.img_size, self.img_size), dtype=np.uint8)

    def load_multimodal_image(self, slice_id: str) -> np.ndarray:
        """Load and stack [t1, t1ce, t2, flair] -> HxWx4 uint8."""
        modalities = ['t1', 't1ce', 't2', 'flair']
        imgs = [self._read_modality(slice_id, m) for m in modalities]
        return np.stack(imgs, axis=-1)

    # ------------------------------
    def load_labels(self, slice_id: str):
        """Load YOLO bboxes and class ids for a slice_id."""
        label_path = None
        for ext in ("txt", "TXT"):
            p = self.label_dir / f"{slice_id}.{ext}"
            if p.exists():
                label_path = p
                break

        bboxes, cls = [], []
        if label_path and label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x, y, w, h = map(float, parts[1:5])
                            # ensure normalized range
                            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0:
                                bboxes.append([x, y, w, h])
                                cls.append(class_id)
            except Exception as e:
                logger.error(f"Error reading label file {label_path}: {e}")

        if not bboxes:
            return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.int64)
        return np.asarray(bboxes, dtype=np.float32), np.asarray(cls, dtype=np.int64)

    # ------------------------------
    def __len__(self):
        return len(self.slice_ids)

    # ------------------------------
    def __getitem__(self, idx):
        slice_id = self.slice_ids[idx]

        image = self.load_multimodal_image(slice_id)            # HxWx4 uint8
        bboxes, class_labels = self.load_labels(slice_id)       # arrays

        try:
            if len(bboxes) > 0:
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes.tolist(),
                    class_labels=class_labels.tolist()
                )
                image = transformed["image"]                    # torch.FloatTensor [4,H,W]
                bboxes = np.asarray(transformed["bboxes"], dtype=np.float32)
                class_labels = np.asarray(transformed["class_labels"], dtype=np.int64)
            else:
                transformed = self.transform(image=image, bboxes=[], class_labels=[])
                image = transformed["image"]
                bboxes = np.zeros((0, 4), dtype=np.float32)
                class_labels = np.array([], dtype=np.int64)

        except Exception as e:
            # Very defensive fallback path: resize + manual z-score + torch
            logger.error(f"Error applying transforms to {slice_id}: {e}")
            img = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA).astype(np.float32)
            # z-score per channel
            mean = img.mean(axis=(0, 1), keepdims=True)
            std = img.std(axis=(0, 1), keepdims=True)
            img = (img - mean) / (std + 1e-6)
            image = torch.from_numpy(img).permute(2, 0, 1).float()
            bboxes = np.zeros((0, 4), dtype=np.float32)
            class_labels = np.array([], dtype=np.int64)

        return {
            "image": image,                 # torch.FloatTensor [4,H,W]
            "bboxes": bboxes,               # np.ndarray [N,4] (YOLO normalized)
            "class_labels": class_labels,   # np.ndarray [N]
            "slice_id": slice_id
        }

# ---------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------
def collate_fn(batch):
    """
    Custom collate: stack images; pad boxes/labels to max per batch with -1 label.
    """
    images = torch.stack([item["image"] for item in batch]).float()  # [B,4,H,W]

    max_boxes = max(len(item["bboxes"]) for item in batch)
    if max_boxes == 0:
        max_boxes = 1

    B = len(batch)
    batch_bboxes = torch.zeros(B, max_boxes, 4, dtype=torch.float32)
    batch_labels = torch.full((B, max_boxes), -1, dtype=torch.long)
    slice_ids = []

    for i, item in enumerate(batch):
        slice_ids.append(item["slice_id"])
        n = len(item["bboxes"])
        if n > 0:
            batch_bboxes[i, :n] = torch.from_numpy(item["bboxes"]).float()
            batch_labels[i, :n] = torch.from_numpy(item["class_labels"]).long()

    return {
        "images": images,       # [B,4,H,W]
        "bboxes": batch_bboxes, # [B,M,4]
        "labels": batch_labels, # [B,M]
        "slice_ids": slice_ids
    }

__all__ = ["BraTSDataset", "collate_fn"]
