"""
Extract 2D radiomics features from every axial slice of each cropped liver volume.

For each series:
  - Load the 3D CT + mask
  - Iterate axial slices where liver coverage >= MIN_LIVER_PX
  - Subsample every STRIDE slices to reduce inter-slice correlation
  - Extract first-order + GLCM features from the liver region of that slice
  - Write one row per slice to CSV

Output: Cropped_Data/slice_radiomics_2d.csv
Columns: series, patient, group, label, slice_idx, liver_px, + features

Run: .venv\Scripts\python.exe extract_2d_slice_features.py
"""

import sys, os; sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")


import os, re
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import stats
from skimage.feature import graycomatrix, graycoprops

BASE_DIR   = r"C:\Users\ronin\PycharmProjects\PFinalproject"
CT_DIR     = os.path.join(BASE_DIR, "Cropped_Data", "All_Series_CT")
MASK_DIR   = os.path.join(BASE_DIR, "Cropped_Data", "All_Series_Masks")
OUT_CSV    = os.path.join(BASE_DIR, "Cropped_Data", "slice_radiomics_2d.csv")

MIN_LIVER_PX = 200   # minimum liver pixels in slice to include it
SLICE_STRIDE = 3     # take every 3rd slice to reduce correlation

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


def extract_first_order_2d(pix):
    hist, _ = np.histogram(pix, bins=32, density=True)
    h_nz = hist[hist > 0]
    entropy = float(-np.sum(h_nz * np.log2(h_nz + 1e-10)))
    return {
        "fo_mean":     float(np.mean(pix)),
        "fo_median":   float(np.median(pix)),
        "fo_std":      float(np.std(pix)),
        "fo_skewness": float(stats.skew(pix)),
        "fo_kurtosis": float(stats.kurtosis(pix)),
        "fo_p10":      float(np.percentile(pix, 10)),
        "fo_p90":      float(np.percentile(pix, 90)),
        "fo_iqr":      float(np.percentile(pix, 75) - np.percentile(pix, 25)),
        "fo_range":    float(pix.max() - pix.min()),
        "fo_entropy":  entropy,
    }


def extract_glcm_2d(sl_hu, mask_2d):
    empty = {k: 0.0 for k in ("glcm_contrast","glcm_correlation","glcm_energy",
                               "glcm_homogeneity","glcm_dissimilarity","glcm_asm")}
    pix = sl_hu[mask_2d > 0]
    if len(pix) < 20 or pix.max() == pix.min():
        return empty
    norm = np.zeros(sl_hu.shape, dtype=np.uint8)
    norm[mask_2d > 0] = ((pix - pix.min()) / (pix.max() - pix.min()) * 63).astype(np.uint8)
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


def extract_slice_features(sl_hu, mask_2d):
    pix = sl_hu[mask_2d > 0]
    feats = extract_first_order_2d(pix)
    feats.update(extract_glcm_2d(sl_hu, mask_2d))
    return feats


if __name__ == "__main__":
    ct_files = sorted(f for f in os.listdir(CT_DIR) if f.endswith(".nii.gz"))
    rows = []

    print(f"Found {len(ct_files)} series\n")

    for ct_fname in ct_files:
        stem = ct_fname.removeprefix("cropped_").removesuffix(".nii.gz")
        m = re.match(r'^(.+)_(s\d+)$', stem)
        if not m:
            continue
        patient_key = m.group(1)
        if patient_key not in PATIENT_META:
            continue

        mask_fname = f"cropped_{stem}_mask.nii.gz"
        ct_path   = os.path.join(CT_DIR,   ct_fname)
        mask_path = os.path.join(MASK_DIR, mask_fname)
        if not os.path.exists(mask_path):
            continue

        ct_3d   = nib.load(ct_path).get_fdata().astype(float)
        mask_3d = nib.load(mask_path).get_fdata().astype(np.uint8)
        n_z     = ct_3d.shape[2]

        meta         = PATIENT_META[patient_key]
        series_rows  = 0

        # Iterate axial slices (axis=2) with stride
        for z in range(0, n_z, SLICE_STRIDE):
            sl   = ct_3d[:, :, z]
            msk  = mask_3d[:, :, z]
            liver_px = int(msk.sum())
            if liver_px < MIN_LIVER_PX:
                continue

            feats = extract_slice_features(sl, msk)
            row   = {
                "series":    stem,
                "patient":   patient_key,
                "group":     meta["group"],
                "label":     meta["label"],
                "slice_idx": z,
                "liver_px":  liver_px,
            }
            row.update(feats)
            rows.append(row)
            series_rows += 1

        print(f"  {stem:<45}  {n_z} slices  ->  {series_rows} used (stride={SLICE_STRIDE})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nTotal: {len(df)} slices  x  {len(df.columns)-6} features")
    print(f"Saved: {OUT_CSV}\n")
    print("Slices per patient:")
    for grp, sub in df.groupby("group"):
        label_s = "Cancer" if sub["label"].iloc[0] == 1 else "Healthy"
        print(f"  {grp:<15} [{label_s}]  {len(sub)} slices")
