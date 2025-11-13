from pathlib import Path
import re, numpy as np, cv2, random

root = Path("data")          # train/val/test live here
split = "test"              # run also for "val" and "test"
images = (root/split/"images")
labels = (root/split/"labels")

pat_img = re.compile(r"^(BraTS20_Training_\d+)_slice_(\d+)_(flair|t1|t1ce|t2)\.png$")
pat_lbl = re.compile(r"^(BraTS20_Training_\d+)_slice_(\d+)\.txt$")

# build per-slice availability across the 4 modalities
by_slice = {}
for p in images.glob("*.png"):
    m = pat_img.match(p.name)
    if not m: continue
    key = (m.group(1), m.group(2))   # (case_id, slice_id)
    by_slice.setdefault(key, set()).add(m.group(3))

stems_img = set(by_slice.keys())
stems_lbl = set()
for p in labels.glob("*.txt"):
    m = pat_lbl.match(p.name)
    if not m: continue
    stems_lbl.add((m.group(1), m.group(2)))

missing_lbl = stems_img - stems_lbl
missing_img = stems_lbl - stems_img
incomplete_modalities = {k:v for k,v in by_slice.items() if len(v) != 4}

print(f"[{split}] slices: {len(stems_img)} | labels: {len(stems_lbl)}")
print(f"  missing labels: {len(missing_lbl)}")
print(f"  missing images: {len(missing_img)}")
print(f"  incomplete modalities: {len(incomplete_modalities)} (≠4)")

# check for black images (any modality completely zero)
def is_black(p):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return im is None or np.max(im) == 0

black = []
for (case_id, sid), mods in by_slice.items():
    for m in mods:
        p = images/f"{case_id}_slice_{sid}_{m}.png"
        if is_black(p):
            black.append((case_id, sid, m))
            break

print(f"  slices with at least one black modality: {len(black)}")
if black[:5]: print("  sample black:", black[:5])
if list(missing_lbl)[:5]: print("  sample missing lbl:", list(missing_lbl)[:5])
if list(missing_img)[:5]: print("  sample missing img:", list(missing_img)[:5])
