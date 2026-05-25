"""
Extract radiomics features from ALL cropped liver series.

Input:
    Cropped_Data/All_Series_CT/    → cropped_<patient>_s<nn>.nii.gz
    Cropped_Data/All_Series_Masks/ → cropped_<patient>_s<nn>_mask.nii.gz

Output:
    Cropped_Data/all_series_radiomics.csv

Features per series (~41 total):
    - First-order statistics (13): mean, median, std, variance, skewness,
      kurtosis, energy, entropy, uniformity, range, p10, p90, iqr
    - Gradient features (3): mean/std/p90 gradient magnitude
    - GLCM texture (18): 6 properties × 3 planes (axial, sagittal, coronal)
    - Shape features (7): volume, extents (z/y/x), elongation, flatness, sphericity

Run: .venv\Scripts\python.exe extract_all_series_features.py
"""

import os
import re
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import stats, ndimage
from skimage.feature import graycomatrix, graycoprops

BASE_DIR   = r"C:\Users\ronin\PycharmProjects\PFinalproject"
CT_DIR     = os.path.join(BASE_DIR, "Cropped_Data", "All_Series_CT")
MASK_DIR   = os.path.join(BASE_DIR, "Cropped_Data", "All_Series_Masks")
OUTPUT_CSV = os.path.join(BASE_DIR, "Cropped_Data", "all_series_radiomics.csv")

# patient_key (from filename) → group + label
PATIENT_META = {
    "Patient_1_After":    {"group": "Patient_1",  "label": 1},
    "Patient_1_Before":   {"group": "Patient_1",  "label": 1},
    "Patient_2_After":    {"group": "Patient_2",  "label": 1},
    "Patient_2_Before":   {"group": "Patient_2",  "label": 1},
    "Patient_KB_Cancer":  {"group": "Patient_KB", "label": 1},
    "Patient_KB_Before":  {"group": "Patient_KB", "label": 1},
    "Patient_GA_Healthy": {"group": "Patient_GA", "label": 0},
    "Patient_VT_Healthy": {"group": "Patient_VT", "label": 0},
}


# ── Feature extraction functions ──────────────────────────────────────────────

def extract_first_order(liver_hu):
    hist, _ = np.histogram(liver_hu, bins=64, density=True)
    h_nz = hist[hist > 0]
    entropy = -np.sum(h_nz * np.log2(h_nz + 1e-10))
    return {
        "fo_mean":       float(np.mean(liver_hu)),
        "fo_median":     float(np.median(liver_hu)),
        "fo_std":        float(np.std(liver_hu)),
        "fo_variance":   float(np.var(liver_hu)),
        "fo_skewness":   float(stats.skew(liver_hu)),
        "fo_kurtosis":   float(stats.kurtosis(liver_hu)),
        "fo_energy":     float(np.sum(liver_hu ** 2)),
        "fo_entropy":    float(entropy),
        "fo_uniformity": float(np.sum(hist ** 2)),
        "fo_range":      float(np.max(liver_hu) - np.min(liver_hu)),
        "fo_p10":        float(np.percentile(liver_hu, 10)),
        "fo_p90":        float(np.percentile(liver_hu, 90)),
        "fo_iqr":        float(np.percentile(liver_hu, 75) - np.percentile(liver_hu, 25)),
    }


def extract_gradient_features(ct_3d, binary_mask):
    gz = ndimage.sobel(ct_3d, axis=0)
    gy = ndimage.sobel(ct_3d, axis=1)
    gx = ndimage.sobel(ct_3d, axis=2)
    grad_mag = np.sqrt(gz**2 + gy**2 + gx**2)
    liver_grad = grad_mag[binary_mask == 1]
    return {
        "grad_mean": float(np.mean(liver_grad)),
        "grad_std":  float(np.std(liver_grad)),
        "grad_p90":  float(np.percentile(liver_grad, 90)),
    }


def extract_glcm(slice_hu, mask_2d):
    empty = {"glcm_contrast": 0., "glcm_correlation": 0., "glcm_energy": 0.,
             "glcm_homogeneity": 0., "glcm_dissimilarity": 0., "glcm_asm": 0.}
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    if not rows.any():
        return empty
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    crop = slice_hu[r0:r1+1, c0:c1+1]
    crop_mask = mask_2d[r0:r1+1, c0:c1+1]
    pix = crop[crop_mask > 0]
    if len(pix) < 10 or pix.max() == pix.min():
        return empty
    norm = np.zeros(crop.shape, dtype=np.uint8)
    norm[crop_mask > 0] = ((pix - pix.min()) / (pix.max() - pix.min()) * 63).astype(np.uint8)
    glcm = graycomatrix(norm, distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=64, symmetric=True, normed=True)
    return {
        "glcm_contrast":      float(graycoprops(glcm, "contrast").mean()),
        "glcm_correlation":   float(graycoprops(glcm, "correlation").mean()),
        "glcm_energy":        float(graycoprops(glcm, "energy").mean()),
        "glcm_homogeneity":   float(graycoprops(glcm, "homogeneity").mean()),
        "glcm_dissimilarity": float(graycoprops(glcm, "dissimilarity").mean()),
        "glcm_asm":           float(graycoprops(glcm, "ASM").mean()),
    }


def extract_shape(binary_mask):
    coords = np.array(np.where(binary_mask))
    ext = coords.max(axis=1) - coords.min(axis=1) + 1
    s = np.sort(ext)
    r = np.max(ext) / 2.0
    return {
        "shape_volume":     float(binary_mask.sum()),
        "shape_extent_z":   float(ext[0]),
        "shape_extent_y":   float(ext[1]),
        "shape_extent_x":   float(ext[2]),
        "shape_elongation": float(s[0] / (s[2] + 1e-10)),
        "shape_flatness":   float(s[0] / (s[1] + 1e-10)),
        "shape_sphericity": float(binary_mask.sum() / ((4/3)*np.pi*r**3 + 1e-10)),
    }


def extract_features(ct_3d, mask_3d):
    """ct_3d already in HU (clipped -100..400). mask_3d is binary (0/1)."""
    liver_hu = ct_3d[mask_3d == 1]
    feats = {}
    feats.update(extract_first_order(liver_hu))
    feats.update(extract_gradient_features(ct_3d, mask_3d))

    # Central planes through liver centroid
    coords = np.array(np.where(mask_3d))
    mid = (coords.min(axis=1) + coords.max(axis=1)) // 2  # (axis0, axis1, axis2)

    planes = [
        ("axial",    ct_3d[:, :, mid[2]], mask_3d[:, :, mid[2]]),   # Z-plane
        ("sagittal", ct_3d[mid[0], :, :], mask_3d[mid[0], :, :]),   # X-plane
        ("coronal",  ct_3d[:, mid[1], :], mask_3d[:, mid[1], :]),   # Y-plane
    ]
    for name, sl, msk in planes:
        for k, v in extract_glcm(sl, msk).items():
            feats[f"{name}_{k}"] = v

    feats.update(extract_shape(mask_3d))
    return feats


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ct_files = sorted(f for f in os.listdir(CT_DIR) if f.endswith(".nii.gz"))
    rows = []

    print(f"Found {len(ct_files)} CT files in {CT_DIR}\n")

    for ct_fname in ct_files:
        # cropped_Patient_1_After_s01.nii.gz → stem = Patient_1_After_s01
        stem = ct_fname.removeprefix("cropped_").removesuffix(".nii.gz")
        m = re.match(r'^(.+)_(s\d+)$', stem)
        if not m:
            print(f"  SKIP (unexpected name): {ct_fname}")
            continue

        patient_key, series_id = m.group(1), m.group(2)
        if patient_key not in PATIENT_META:
            print(f"  SKIP (unknown patient key): {patient_key}")
            continue

        mask_fname = f"cropped_{stem}_mask.nii.gz"
        ct_path    = os.path.join(CT_DIR,   ct_fname)
        mask_path  = os.path.join(MASK_DIR, mask_fname)

        if not os.path.exists(mask_path):
            print(f"  SKIP (no mask): {mask_fname}")
            continue

        print(f"  {stem:<40}", end=" ", flush=True)

        ct_3d   = nib.load(ct_path).get_fdata().astype(float)
        mask_3d = nib.load(mask_path).get_fdata().astype(np.uint8)

        if mask_3d.sum() == 0:
            print("no liver voxels — skipped")
            continue

        meta  = PATIENT_META[patient_key]
        feats = extract_features(ct_3d, mask_3d)

        row = {
            "series":  stem,
            "patient": patient_key,
            "group":   meta["group"],
            "label":   meta["label"],
        }
        row.update(feats)
        rows.append(row)
        print(f"shape={ct_3d.shape}  liver={mask_3d.sum()} voxels  ->  {len(feats)} features")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} series x {len(df.columns)-4} features  ->  {OUTPUT_CSV}")

    # Summary
    print("\nPer-patient series count:")
    for grp, sub in df.groupby("group"):
        label_str = "Cancer" if sub["label"].iloc[0] == 1 else "Healthy"
        print(f"  {grp:<15} [{label_str}]  {len(sub)} series")
