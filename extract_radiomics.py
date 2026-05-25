"""
Extract 3D radiomics features from liver CT volumes using scikit-image + scipy.

Uses:
  - CT from Nifti_Volumes/ (raw uint16, stored as HU+1024, no clipping applied)
  - Masks from Liver_Masks/ (liver label = 5 from TotalSegmentator)

Features extracted per patient:
  - First-order statistics (13 features) from all 3D liver voxels
  - GLCM texture features (6 features x 3 planes) — cropped to liver bounding box
  - Gradient features (3 features): mean/std/max gradient magnitude (heterogeneity)
  - Shape features (7 features) from the 3D binary mask
Total: ~41 features
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import stats, ndimage
from skimage.feature import graycomatrix, graycoprops

CT_DIR = r"c:\Users\ronin\PycharmProjects\PFinalproject\Nifti_Volumes"
MASK_DIR = r"c:\Users\ronin\PycharmProjects\PFinalproject\Liver_Masks"
OUTPUT_CSV = r"c:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\liver_radiomics.csv"

LIVER_LABEL = 5  # TotalSegmentator assigns label 5 to liver
HU_OFFSET = 1024  # raw values are stored as HU + 1024

# All 8 scans — patient-level LOOCV prevents leakage (no scan-level train/test split)
ALL_PATIENTS = {
    "Patient_1_Before_CT":   {"label": 1, "group": "Patient_1"},
    "Patient_1_After_CT":    {"label": 1, "group": "Patient_1"},
    "Patient_2_Before_CT":   {"label": 1, "group": "Patient_2"},
    "Patient_2_After_CT":    {"label": 1, "group": "Patient_2"},
    "Patient_KB_Before_CT":  {"label": 1, "group": "Patient_KB"},
    "Patient_KB_Cancer_CT":  {"label": 1, "group": "Patient_KB"},
    "Patient_GA_Healthy_CT": {"label": 0, "group": "Patient_GA"},
    "Patient_VT_Healthy_CT": {"label": 0, "group": "Patient_VT"},
}

# ICC pairs: same patient measured twice (before → after cancer development)
ICC_PAIRS = [
    ("Patient_1_Before_CT",  "Patient_1_After_CT"),
    ("Patient_2_Before_CT",  "Patient_2_After_CT"),
    ("Patient_KB_Before_CT", "Patient_KB_Cancer_CT"),
]


def to_hu(voxels):
    return voxels.astype(float) - HU_OFFSET


def extract_first_order(liver_hu):
    hist, _ = np.histogram(liver_hu, bins=64, density=True)
    hist_nonzero = hist[hist > 0]
    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero + 1e-10))
    uniformity = np.sum(hist ** 2)

    return {
        "fo_mean": float(np.mean(liver_hu)),
        "fo_median": float(np.median(liver_hu)),
        "fo_std": float(np.std(liver_hu)),
        "fo_variance": float(np.var(liver_hu)),
        "fo_skewness": float(stats.skew(liver_hu)),
        "fo_kurtosis": float(stats.kurtosis(liver_hu)),
        "fo_energy": float(np.sum(liver_hu ** 2)),
        "fo_entropy": float(entropy),
        "fo_uniformity": float(uniformity),
        "fo_range": float(np.max(liver_hu) - np.min(liver_hu)),
        "fo_p10": float(np.percentile(liver_hu, 10)),
        "fo_p90": float(np.percentile(liver_hu, 90)),
        "fo_iqr": float(np.percentile(liver_hu, 75) - np.percentile(liver_hu, 25)),
    }


def extract_gradient_features(ct_3d, binary_mask):
    """Mean/std/max gradient magnitude inside the liver — measures heterogeneity."""
    gz = ndimage.sobel(ct_3d, axis=0)
    gy = ndimage.sobel(ct_3d, axis=1)
    gx = ndimage.sobel(ct_3d, axis=2)
    grad_mag = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)
    liver_grad = grad_mag[binary_mask == 1]
    return {
        "grad_mean": float(np.mean(liver_grad)),
        "grad_std": float(np.std(liver_grad)),
        "grad_p90": float(np.percentile(liver_grad, 90)),
    }


def extract_glcm_on_crop(slice_hu, mask_2d):
    """GLCM computed only on the liver bounding box — excludes background zeros."""
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    if not rows.any():
        return {k: 0.0 for k in [
            "glcm_contrast", "glcm_correlation", "glcm_energy",
            "glcm_homogeneity", "glcm_dissimilarity", "glcm_asm"
        ]}

    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]

    cropped = slice_hu[r0:r1 + 1, c0:c1 + 1]
    cropped_mask = mask_2d[r0:r1 + 1, c0:c1 + 1]
    liver_pix = cropped[cropped_mask > 0]

    if len(liver_pix) < 10:
        return {k: 0.0 for k in [
            "glcm_contrast", "glcm_correlation", "glcm_energy",
            "glcm_homogeneity", "glcm_dissimilarity", "glcm_asm"
        ]}

    s_min, s_max = liver_pix.min(), liver_pix.max()
    if s_max == s_min:
        return {k: 0.0 for k in [
            "glcm_contrast", "glcm_correlation", "glcm_energy",
            "glcm_homogeneity", "glcm_dissimilarity", "glcm_asm"
        ]}

    # Normalize only liver pixels; background stays 0 but is inside bounding box
    normalized = np.zeros(cropped.shape, dtype=np.uint8)
    normalized[cropped_mask > 0] = (
        (liver_pix - s_min) / (s_max - s_min) * 63
    ).astype(np.uint8)

    glcm = graycomatrix(
        normalized, distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=64, symmetric=True, normed=True
    )
    return {
        "glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
        "glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
        "glcm_energy": float(graycoprops(glcm, "energy").mean()),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "glcm_dissimilarity": float(graycoprops(glcm, "dissimilarity").mean()),
        "glcm_asm": float(graycoprops(glcm, "ASM").mean()),
    }


def extract_shape_features(binary_mask):
    coords = np.array(np.where(binary_mask))
    extents = coords.max(axis=1) - coords.min(axis=1) + 1
    sorted_ext = np.sort(extents)
    r = np.max(extents) / 2.0
    sphere_vol = (4 / 3) * np.pi * r ** 3
    return {
        "shape_volume": float(np.sum(binary_mask)),
        "shape_extent_z": float(extents[0]),
        "shape_extent_y": float(extents[1]),
        "shape_extent_x": float(extents[2]),
        "shape_elongation": float(sorted_ext[0] / (sorted_ext[2] + 1e-10)),
        "shape_flatness": float(sorted_ext[0] / (sorted_ext[1] + 1e-10)),
        "shape_sphericity": float(np.sum(binary_mask) / (sphere_vol + 1e-10)),
    }


def extract_patient_features(patient_name, label):
    ct_path = os.path.join(CT_DIR, f"{patient_name}.nii.gz")
    mask_path = os.path.join(MASK_DIR, f"{patient_name}_mask.nii.gz")

    if not os.path.exists(ct_path):
        print(f"WARNING: CT not found: {ct_path}")
        return None
    if not os.path.exists(mask_path):
        print(f"WARNING: Mask not found: {mask_path}")
        return None

    ct_raw = nib.load(ct_path).get_fdata()
    mask_data = nib.load(mask_path).get_fdata()
    binary_mask = (mask_data == LIVER_LABEL).astype(np.uint8)

    if binary_mask.sum() == 0:
        print(f"WARNING: Empty liver mask for {patient_name}")
        return None

    print(f"  {patient_name}: shape {ct_raw.shape}, liver voxels {binary_mask.sum()}")

    ct_hu = to_hu(ct_raw)
    liver_hu = ct_hu[binary_mask == 1]

    features = {"patient": patient_name, "label": label}
    features.update(extract_first_order(liver_hu))
    features.update(extract_gradient_features(ct_hu, binary_mask))

    # GLCM from 3 orthogonal planes through liver centroid, cropped to bounding box
    liver_coords = np.array(np.where(binary_mask))
    mid_z = int((liver_coords[0].min() + liver_coords[0].max()) // 2)
    mid_y = int((liver_coords[1].min() + liver_coords[1].max()) // 2)
    mid_x = int((liver_coords[2].min() + liver_coords[2].max()) // 2)

    planes = [
        ("axial",    ct_hu[mid_z, :, :],   binary_mask[mid_z, :, :]),
        ("sagittal", ct_hu[:, :, mid_x],   binary_mask[:, :, mid_x]),
        ("coronal",  ct_hu[:, mid_y, :],   binary_mask[:, mid_y, :]),
    ]
    for plane_name, slice_2d, mask_2d in planes:
        for k, v in extract_glcm_on_crop(slice_2d, mask_2d).items():
            features[f"{plane_name}_{k}"] = v

    features.update(extract_shape_features(binary_mask))
    return features


if __name__ == "__main__":
    print("Extracting 3D radiomics features (HU-corrected, GLCM cropped to liver)...")
    rows = []
    for patient_name, info in ALL_PATIENTS.items():
        label, group = info["label"], info["group"]
        print(f"\nProcessing {patient_name} (label={label}, group={group})...")
        row = extract_patient_features(patient_name, label)
        if row is not None:
            row["group"] = group
            rows.append(row)
            print(f"  -> {len(row) - 3} features extracted")

    if not rows:
        print("ERROR: No features extracted.")
        exit(1)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} patients x {len(df.columns) - 3} features -> {OUTPUT_CSV}")
    print(f"  Groups: {df['group'].unique().tolist()}")

    feature_cols = [c for c in df.columns if c not in ("patient", "label")]
    print("\nPer-patient values for key features:")
    key_features = ["fo_mean", "fo_std", "fo_uniformity", "grad_mean", "fo_entropy",
                    "axial_glcm_contrast", "axial_glcm_homogeneity"]
    for feat in key_features:
        if feat in df.columns:
            print(f"\n  {feat}:")
            for _, row in df.iterrows():
                label_str = "CANCER" if row["label"] == 1 else "HEALTHY"
                print(f"    {row['patient']:35s} [{label_str}]: {row[feat]:.4f}")
