"""
Full pipeline for all DICOM series per patient:
  Step 1 — Save every valid series as NIfTI  →  All_Series_Nifti/
  Step 2 — Run TotalSegmentator (fast) on each  →  All_Series_Masks/
  Step 3 — Visualize: one PNG per patient, all series with liver overlay

Run: .venv\Scripts\python.exe segment_all_series.py
"""

import os
import sys
import subprocess
import numpy as np
import pydicom
import nibabel as nib
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

TS_CLI = os.path.join(os.path.dirname(sys.executable), "TotalSegmentator.exe")

BASE_DIR   = r"C:\Users\ronin\PycharmProjects\PFinalproject"
NIFTI_DIR  = os.path.join(BASE_DIR, "All_Series_Nifti")
MASK_DIR   = os.path.join(BASE_DIR, "All_Series_Masks")
OUT_DIR    = os.path.join(BASE_DIR, "series_per_patient")
HU_OFFSET  = 1024
LIVER_LABEL = 5

for d in (NIFTI_DIR, MASK_DIR, OUT_DIR):
    os.makedirs(d, exist_ok=True)

SKIP_DIRS = {'.venv', 'venv', 'Nifti_Volumes', 'Liver_Masks', 'All_Series_Nifti',
             'All_Series_Masks', 'Cropped_Data', '__pycache__', '.git', '.claude',
             '.idea', 'finalProject', 'docs', 'series_per_patient'}

TRANSLATION = {
    "פציינט 1 אחרי סרטן":   ("Patient_1_After",    "Patient 1 — After Cancer",   1),
    "מטופל G-A ללא סרטן":   ("Patient_GA_Healthy",  "Patient GA — Healthy",        0),
    "מטופל K-B לפני סרטן":  ("Patient_KB_Before",   "Patient KB — Before Cancer",  1),
    "מטופל K-B עם סרטן":    ("Patient_KB_Cancer",   "Patient KB — Cancer",         1),
    "מטופל V-T ללא סרטן":   ("Patient_VT_Healthy",  "Patient VT — Healthy",        0),
    "מטופל 1 לפני 3 פאזות": ("Patient_1_Before",    "Patient 1 — Before (phases)", 1),
    "מטופל 2 אחרי סרטן":    ("Patient_2_After",     "Patient 2 — After Cancer",    1),
    "מטופל 2 לפני סרטן":    ("Patient_2_Before",    "Patient 2 — Before Cancer",   1),
}

COLOR_CANCER  = "#C62828"
COLOR_HEALTHY = "#1B5E20"


def window(arr, center=60, width=400):
    lo, hi = center - width / 2, center + width / 2
    return np.clip(arr.astype(float), lo, hi)


def best_z(liver_mask):
    return int(np.argmax(liver_mask.sum(axis=(0, 1))))


def read_valid_series(folder_path):
    """Return list of (desc, volume_ZYX, n_slices, z_positions)."""
    series_dict = defaultdict(list)
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            try:
                dcm = pydicom.dcmread(os.path.join(root, fname), force=True)
                if hasattr(dcm, 'pixel_array'):
                    uid  = dcm.get('SeriesInstanceUID', 'unknown')
                    desc = str(dcm.get('SeriesDescription', '')).strip() or uid[-8:]
                    series_dict[uid].append((dcm, desc))
            except Exception:
                pass

    valid = []
    for uid, items in series_dict.items():
        slices = [d for d, _ in items]
        desc   = items[0][1]
        if len(slices) < 10:
            continue
        shapes = {s.pixel_array.shape for s in slices}
        if len(shapes) > 1 or next(iter(shapes))[1] != 512:
            continue
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            volume = np.array([s.pixel_array for s in slices])
            z_pos  = [float(s.ImagePositionPatient[2]) for s in slices]
            px_sp  = slices[0].PixelSpacing
            z_sp   = float(slices[0].SliceThickness)
            valid.append((desc, volume, len(slices), z_pos, float(px_sp[0]), float(px_sp[1]), z_sp))
        except Exception:
            pass

    valid.sort(key=lambda x: x[2], reverse=True)
    return valid


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1+2: Save NIfTI + Run TotalSegmentator for every series
# ══════════════════════════════════════════════════════════════════════════════

patient_folders = sorted([
    f for f in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, f)) and f not in SKIP_DIRS
])

# Collect all series info for visualization later
all_patients_data = {}   # key → (display, label, [(desc, nifti_path, mask_path, n_slices), ...])

for folder_name in patient_folders:
    if folder_name not in TRANSLATION:
        continue
    key, display, label = TRANSLATION[folder_name]
    print(f"\n{'='*60}")
    print(f"Patient: {display}  [{'Cancer' if label==1 else 'Healthy'}]")
    print(f"{'='*60}")

    series_list = read_valid_series(os.path.join(BASE_DIR, folder_name))
    if not series_list:
        print("  No valid series found.")
        continue

    patient_series = []

    for idx, (desc, volume, n_slices, z_pos, px_x, px_y, z_sp) in enumerate(series_list, 1):
        stem      = f"{key}_s{idx:02d}"
        nifti_path = os.path.join(NIFTI_DIR, f"{stem}.nii.gz")
        mask_path  = os.path.join(MASK_DIR,  f"{stem}_mask.nii.gz")

        # ── Save NIfTI (skip if already exists) ───────────────────────────────
        if not os.path.exists(nifti_path):
            print(f"  [{idx}] {desc} ({n_slices} slices) — saving NIfTI...")
            sitk_img = sitk.GetImageFromArray(volume)
            sitk_img.SetSpacing((px_x, px_y, z_sp))
            sitk.WriteImage(sitk_img, nifti_path)
        else:
            print(f"  [{idx}] {desc} ({n_slices} slices) — NIfTI exists, skipping save")

        # ── Run TotalSegmentator CLI (skip if mask already exists) ────────────
        if not os.path.exists(mask_path):
            print(f"       Running TotalSegmentator (fast mode)...", flush=True)
            try:
                result = subprocess.run(
                    [TS_CLI, "-i", nifti_path, "-o", mask_path, "-ml", "-f"],
                    capture_output=True, text=True
                )
                if result.returncode == 0 and os.path.exists(mask_path):
                    print(f"       Mask saved.", flush=True)
                else:
                    print(f"       ERROR (rc={result.returncode}): {result.stderr[-300:]}", flush=True)
                    mask_path = None
            except Exception as e:
                print(f"       ERROR: {e}", flush=True)
                mask_path = None
        else:
            print(f"       Mask exists, skipping segmentation", flush=True)

        patient_series.append((desc, nifti_path, mask_path, n_slices))

    all_patients_data[key] = (display, label, patient_series)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Visualize — one PNG per patient
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("Creating visualizations...")
print(f"{'='*60}")

for key, (display, label, patient_series) in all_patients_data.items():
    n_series = len(patient_series)
    if n_series == 0:
        continue

    print(f"  Visualizing {key}...", flush=True)
    row_color = COLOR_CANCER if label == 1 else COLOR_HEALTHY
    label_str = "Cancer" if label == 1 else "Healthy"

    try:
        fig, axes = plt.subplots(1, n_series, figsize=(n_series * 4.5, 5.8))
    except Exception as e:
        print(f"  ERROR creating figure for {key}: {e}", flush=True)
        continue
    if n_series == 1:
        axes = [axes]

    fig.suptitle(
        f"{display}  [{label_str}]\n"
        f"Red/orange overlay = liver (TotalSegmentator)",
        fontsize=12, fontweight="bold", color=row_color, y=1.04
    )

    best_n = max(s[3] for s in patient_series)

    for col, (desc, nifti_path, mask_path, n_slices) in enumerate(patient_series):
        ax = axes[col]
        is_best = (n_slices == best_n)

        # Load CT
        try:
            ct_data = nib.load(nifti_path).get_fdata()
            # Spectral CT series may have extra dimensions — keep only first 3D volume
            while ct_data.ndim > 3:
                ct_data = ct_data[..., 0]
        except Exception as e:
            ax.text(0.5, 0.5, f"Load error:\n{e}", ha='center', va='center',
                    transform=ax.transAxes, fontsize=7, color='red')
            ax.axis('off')
            continue
        ct_hu   = ct_data.astype(float) - HU_OFFSET

        # Load liver mask
        liver_mask = None
        if mask_path and os.path.exists(mask_path):
            seg_data   = nib.load(mask_path).get_fdata()
            liver_mask = (seg_data == LIVER_LABEL).astype(np.uint8)

        # Pick best liver slice or middle
        if liver_mask is not None and liver_mask.sum() > 0:
            iz = best_z(liver_mask)
        else:
            iz = ct_hu.shape[2] // 2 if ct_hu.ndim == 3 else ct_hu.shape[0] // 2

        # Get the axial CT slice
        if ct_hu.ndim == 3:
            # NIfTI from SimpleITK: shape might be (X, Y, Z)
            ct_sl = window(ct_hu[:, :, iz]).T   # → display as (Y, X)
        else:
            ct_sl = window(ct_hu)

        ax.imshow(ct_sl, cmap="gray", origin="lower", aspect="auto")

        # Liver overlay
        if liver_mask is not None and liver_mask.sum() > 0:
            if liver_mask.ndim == 3:
                iz_mask = min(iz, liver_mask.shape[2] - 1)
                mask_sl = liver_mask[:, :, iz_mask].T  # (X,Y) → (Y,X)
            else:
                mask_sl = liver_mask
            overlay = np.ma.masked_where(mask_sl == 0, mask_sl.astype(float))
            ax.imshow(overlay, cmap="autumn", alpha=0.65,
                      origin="lower", aspect="auto", vmin=0, vmax=1)

        # Border: gold for best/most slices, white for others
        bcolor = "#FFD600" if is_best else "#AAAAAA"
        blw    = 3.5       if is_best else 1.2
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(bcolor)
            spine.set_linewidth(blw)

        if is_best:
            ax.text(0.97, 0.03, "SELECTED\nfor analysis", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.5, fontweight="bold",
                    color="#FFD600",
                    bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.75))

        no_mask_note = "" if (mask_path and os.path.exists(str(mask_path))) else "\n[no mask]"
        ax.set_title(f"{desc}\n{n_slices} slices{no_mask_note}",
                     fontsize=8, pad=4,
                     color="#FFD600" if is_best else "#CCCCCC",
                     fontweight="bold" if is_best else "normal")
        ax.axis("off")

    # Legend
    patches = [
        mpatches.Patch(color="#FFD600",  label="Selected for ML analysis (most slices)"),
        mpatches.Patch(color="#FF4500", alpha=0.7, label="Liver segmentation (TotalSegmentator)"),
        mpatches.Patch(color="#AAAAAA", label="Other series"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.07))

    try:
        plt.tight_layout()
    except Exception:
        pass
    out_path = os.path.join(OUT_DIR, f"{key}.png")
    try:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}", flush=True)
    except Exception as e:
        print(f"  ERROR saving {key}: {e}", flush=True)
    plt.close(fig)

print(f"\nDone. PNGs in: {OUT_DIR}")
