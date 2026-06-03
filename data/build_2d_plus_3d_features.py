"""
Join 3D series features onto each 2D slice row.

Problem with pure 2D: a single axial slice cannot see sagittal/coronal texture.
sagittal_glcm_dissimilarity (the top VaRFS feature) requires the full 3D volume.

Solution: for every slice, append the 3D features of its parent series.
Each slice ends up with:
  - 16 axial 2D features  (slice-specific: what this slice looks like)
  - 38 3D volume features  (series-level: what the whole liver looks like)
  = 54 features total

The 3D features are CONSTANT across all slices of a series.
This lets the classifier use sagittal_glcm_dissimilarity (and shape, gradient, etc.)
while still having per-slice variation from the axial features.

Output: Cropped_Data/slice_2d_plus_3d.csv

Run: .venv\Scripts\python.exe data\build_2d_plus_3d_features.py
"""
import sys, os
sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")

import pandas as pd

CSV_2D  = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\slice_radiomics_2d.csv"
CSV_3D  = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\all_series_radiomics.csv"
OUT_CSV = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\slice_2d_plus_3d.csv"

METADATA_COLS = {"series", "patient", "group", "label", "slice_idx", "liver_px"}

# ── Load ──────────────────────────────────────────────────────────────────────

df_2d = pd.read_csv(CSV_2D)
df_3d = pd.read_csv(CSV_3D)

feat_2d = [c for c in df_2d.columns if c not in METADATA_COLS]
feat_3d = [c for c in df_3d.columns if c not in {"series", "patient", "group", "label"}]

print(f"2D CSV: {len(df_2d)} slices x {len(feat_2d)} features")
print(f"3D CSV: {len(df_3d)} series x {len(feat_3d)} features")

# ── Find overlapping feature names ────────────────────────────────────────────
# Both CSVs share first-order names (fo_mean, fo_kurtosis, etc.)
# Rename 3D versions with vol_ prefix to avoid collision

overlap = set(feat_2d) & set(feat_3d)
print(f"\nOverlapping feature names (will be prefixed vol_ in 3D): {sorted(overlap)}")

df_3d_renamed = df_3d[["series"] + feat_3d].copy()
rename_map = {f: f"vol_{f}" for f in feat_3d if f in overlap}
df_3d_renamed = df_3d_renamed.rename(columns=rename_map)

# ── Merge on series ───────────────────────────────────────────────────────────
df = pd.merge(df_2d, df_3d_renamed, on="series", how="left")

# Check for unmatched slices
n_missing = df[[f"vol_{f}" if f in overlap else f for f in feat_3d[:3]]].isna().any(axis=1).sum()
if n_missing > 0:
    print(f"\nWARNING: {n_missing} slices had no matching 3D series — check series names")
else:
    print(f"\nAll slices matched to a 3D series. OK.")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(OUT_CSV, index=False)

all_feat_cols = [c for c in df.columns if c not in METADATA_COLS]
print(f"\nOutput: {len(df)} slices x {len(all_feat_cols)} features")
print(f"  2D features (slice-specific): {len(feat_2d)}")
print(f"  3D features (series-level):   {len(feat_3d)}")
print(f"  Total:                        {len(all_feat_cols)}")
print(f"\nSaved: {OUT_CSV}")

print("\nSample columns:")
for c in all_feat_cols[:10]:
    print(f"  {c}")
print("  ...")
for c in all_feat_cols[-5:]:
    print(f"  {c}")
