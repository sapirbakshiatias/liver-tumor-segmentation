"""
Inspect what's inside the gray (NaN) areas of spatial heatmaps.

For a chosen series, prints:
  - How many pixels are gray and WHY (outside mask vs edge NaN vs stride gap)
  - HU statistics of mask-edge pixels (the gray band at liver border)
  - Feature value range vs the NaN regions
"""
import sys, warnings
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")
warnings.filterwarnings("ignore")

from train_all_series_report import (
    sliding_feature_map, load_best_axial_slice, OUT_DIR,
)
import os

WINDOW = 16
STRIDE = 4
SERIES = "Patient_KB_Cancer_s01"   # change to inspect another series

ct_sl, msk_sl = load_best_axial_slice(SERIES)
if ct_sl is None:
    print("Series not found"); sys.exit(1)

h, w = ct_sl.shape
print(f"Slice shape: {h} x {w}")
print(f"Liver pixels in slice: {msk_sl.sum()}  ({100*msk_sl.sum()/(h*w):.1f}% of frame)")

# ── Categorise every pixel ────────────────────────────────────────────────────
outside_mask  = msk_sl == 0              # gray reason 1: outside segmentation
inside_mask   = msk_sl == 1

# Which inside-mask pixels are never covered by any window centre?
covered = np.zeros((h, w), dtype=bool)
for y0 in range(0, h - WINDOW + 1, STRIDE):
    for x0 in range(0, w - WINDOW + 1, STRIDE):
        pmask = msk_sl[y0:y0+WINDOW, x0:x0+WINDOW]
        if pmask.sum() >= (WINDOW // 2):
            covered[y0:y0+WINDOW, x0:x0+WINDOW] = True

uncovered_inside = inside_mask & ~covered   # gray reason 2+3: NaN inside liver

print(f"\nPixel breakdown:")
print(f"  Outside mask (background):          {outside_mask.sum():>6}  ({100*outside_mask.sum()/(h*w):.1f}%)")
print(f"  Inside mask, covered by heatmap:    {(inside_mask & covered).sum():>6}  ({100*(inside_mask & covered).sum()/(h*w):.1f}%)")
print(f"  Inside mask, NOT covered (gray):    {uncovered_inside.sum():>6}  ({100*uncovered_inside.sum()/(h*w):.1f}%)")

# ── HU values in uncovered region ─────────────────────────────────────────────
if uncovered_inside.sum() > 0:
    hu_uncov = ct_sl[uncovered_inside]
    hu_all   = ct_sl[inside_mask]
    print(f"\nHU values — uncovered liver edge pixels:")
    print(f"  Mean:   {hu_uncov.mean():.1f}  (full liver mean: {hu_all.mean():.1f})")
    print(f"  Median: {np.median(hu_uncov):.1f}")
    print(f"  Range:  [{hu_uncov.min():.1f}, {hu_uncov.max():.1f}]")
    print(f"  Interpretation: {'similar to liver' if abs(hu_uncov.mean()-hu_all.mean())<30 else 'DIFFERENT from liver core — possible vessel/lesion edge'}")

# ── Feature map and NaN analysis ──────────────────────────────────────────────
feat_name = "fo_mean"
fmap = sliding_feature_map(ct_sl, msk_sl, feat_name, win=WINDOW, stride=STRIDE)

nan_inside  = inside_mask & np.isnan(fmap)
valid_inside = inside_mask & ~np.isnan(fmap)

print(f"\nFeature map '{feat_name}':")
print(f"  Valid (colored) liver pixels:   {valid_inside.sum():>6}")
print(f"  NaN (gray) liver pixels:        {nan_inside.sum():>6}")
print(f"  Valid value range: [{fmap[valid_inside].min():.2f}, {fmap[valid_inside].max():.2f}]")
if nan_inside.sum() > 0:
    hu_nan = ct_sl[nan_inside]
    print(f"  HU at NaN pixels: mean={hu_nan.mean():.1f}  range=[{hu_nan.min():.1f},{hu_nan.max():.1f}]")

# ── Diagnostic figure ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ct_disp = np.clip(ct_sl, -100, 400)
ct_disp = (ct_disp - ct_disp.min()) / (ct_disp.max() - ct_disp.min() + 1e-10)

# Panel 1: CT + mask outline
axes[0].imshow(ct_disp, cmap="gray", origin="lower", aspect="auto")
axes[0].contour(msk_sl, levels=[0.5], colors=["lime"], linewidths=1.5)
axes[0].set_title("CT + Mask boundary (green)", fontweight="bold")
axes[0].axis("off")

# Panel 2: Coverage map
coverage_map = np.zeros((h, w, 3), dtype=float)  # RGB
coverage_map[outside_mask]    = [0.2, 0.2, 0.2]   # dark gray = outside mask
coverage_map[inside_mask & covered] = [0.2, 0.8, 0.2]   # green = computed
coverage_map[uncovered_inside] = [1.0, 0.4, 0.0]   # orange = gray holes in liver

axes[1].imshow(coverage_map, origin="lower", aspect="auto")
patches = [
    mpatches.Patch(color=[0.2,0.2,0.2], label="Outside mask"),
    mpatches.Patch(color=[0.2,0.8,0.2], label="Computed (colored)"),
    mpatches.Patch(color=[1.0,0.4,0.0], label="Gray holes (NaN inside liver)"),
]
axes[1].legend(handles=patches, fontsize=7, loc="upper right")
axes[1].set_title("Why pixels are gray", fontweight="bold")
axes[1].axis("off")

# Panel 3: actual heatmap
fmap_masked = np.ma.masked_where(msk_sl == 0, fmap)
axes[2].imshow(ct_disp, cmap="gray", origin="lower", aspect="auto")
im = axes[2].imshow(fmap_masked, cmap="hot", alpha=0.75, origin="lower", aspect="auto")
plt.colorbar(im, ax=axes[2], fraction=0.04)
axes[2].set_title(f"Feature map: {feat_name}", fontweight="bold")
axes[2].axis("off")

plt.suptitle(f"Gray area inspection — {SERIES}", fontsize=12, fontweight="bold")
plt.tight_layout()
out = os.path.join(OUT_DIR, "gray_area_inspection.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")
