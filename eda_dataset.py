import os
import hashlib
from collections import Counter
from PIL import Image
import numpy as np

base_dir = r"c:\Users\rravi\Downloads\Orange Dataset"
splits = ['train', 'test']

print("="*60)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("="*60)

hashes = {}
duplicates = []
corrupted = []
dimensions = []
modes = Counter()
class_counts = {}

for split in splits:
    split_dir = os.path.join(base_dir, split)
    class_counts[split] = {}
    if not os.path.exists(split_dir):
        continue
    for cls in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = os.listdir(cls_dir)
        valid_files = 0
        for f in files:
            fpath = os.path.join(cls_dir, f)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, 'rb') as fp:
                    content = fp.read()
                    h = hashlib.md5(content).hexdigest()
                    if h in hashes:
                        duplicates.append((fpath, hashes[h]))
                    else:
                        hashes[h] = fpath
                with Image.open(fpath) as img:
                    modes[img.mode] += 1
                    dimensions.append(img.size)
                    valid_files += 1
            except Exception as e:
                corrupted.append((fpath, str(e)))
        class_counts[split][cls] = valid_files

print("\n--- Class Counts ---")
for split, counts in class_counts.items():
    print(f"[{split.upper()}] Total: {sum(counts.values())}")
    for cls, cnt in counts.items():
        pct = (cnt / sum(counts.values())) * 100 if sum(counts.values()) > 0 else 0
        print(f"  - {cls:<12}: {cnt:4d} ({pct:5.1f}%)")

print("\n--- Image Color Modes ---")
for mode, cnt in modes.items():
    print(f"  - {mode}: {cnt}")

print("\n--- Image Dimensions ---")
widths = [d[0] for d in dimensions]
heights = [d[1] for d in dimensions]
print(f"  - Min width x height: {min(widths)} x {min(heights)}")
print(f"  - Max width x height: {max(widths)} x {max(heights)}")
print(f"  - Mean width x height: {np.mean(widths):.1f} x {np.mean(heights):.1f}")
print(f"  - Unique aspect ratios: {len(set(round(w/h, 2) for w, h in dimensions))}")

print(f"\n--- Data Quality Issues ---")
print(f"  - Exact Duplicate Images: {len(duplicates)}")
if duplicates:
    print(f"    Sample duplicate: {duplicates[0][0]} == {duplicates[0][1]}")
print(f"  - Corrupted Images: {len(corrupted)}")

print("\nDone EDA inspection.")
