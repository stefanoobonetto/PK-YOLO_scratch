import os
import re
from pathlib import Path
import shutil

root_dir = Path("data")  # directory principale con train/val/test
slice_min_keep = 34
slice_max_keep = 129

pattern_img = re.compile(r"(BraTS20_Training_\d+)_slice_(\d+)_(flair|t1|t1ce|t2)\.png$")
pattern_lbl = re.compile(r"(BraTS20_Training_\d+)_slice_(\d+)\.txt$")


def renumber_and_clean(split_dir):
    """Crea nuove cartelle con file rinumerati e puliti."""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_out = split_dir / "images_clean"
    labels_out = split_dir / "labels_clean"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"⚠️  Mancano images/labels in {split_dir}")
        return

    # Pulisci eventuali directory vecchie
    if images_out.exists():
        shutil.rmtree(images_out)
    if labels_out.exists():
        shutil.rmtree(labels_out)

    images_out.mkdir(exist_ok=True)
    labels_out.mkdir(exist_ok=True)

    print(f"\n📂 Elaboro split: {split_dir.name}")

    image_files = sorted(images_dir.glob("*.png"))

    # Raggruppa per sample
    samples = {}
    for img_path in image_files:
        match = pattern_img.match(img_path.name)
        if not match:
            continue
        prefix, slice_num, modality = match.groups()
        slice_num = int(slice_num)
        samples.setdefault(prefix, []).append((slice_num, modality, img_path))

    for prefix, entries in samples.items():
        valid_slices = [(s, m, p) for (s, m, p) in entries if slice_min_keep <= s <= slice_max_keep]
        valid_slices.sort(key=lambda x: x[0])

        print(f"  {prefix}: tengo {len(valid_slices)} slice")

        for new_idx, (old_idx, modality, path_img) in enumerate(valid_slices):
            new_name = f"{prefix}_slice_{new_idx:03d}_{modality}.png"
            shutil.copy2(path_img, images_out / new_name)

    # --- Labels ---
    label_files = sorted(labels_dir.glob("*.txt"))
    for lbl_path in label_files:
        match = pattern_lbl.match(lbl_path.name)
        if not match:
            continue
        prefix, slice_num = match.groups()
        slice_num = int(slice_num)

        if slice_min_keep <= slice_num <= slice_max_keep:
            new_idx = slice_num - slice_min_keep
            new_name = f"{prefix}_slice_{new_idx:03d}.txt"
            shutil.copy2(lbl_path, labels_out / new_name)

    print(f"✅ Creato split pulito in: {images_out} / {labels_out}")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        split_dir = root_dir / split
        if split_dir.exists():
            renumber_and_clean(split_dir)
        else:
            print(f"❌ Split {split} non trovato!")
