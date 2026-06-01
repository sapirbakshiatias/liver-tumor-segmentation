"""
2D liver views (axial / coronal / sagittal) for ALL series per patient.
One PNG per patient saved to all_series_2d/
"""

import os
import re
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

BASE_DIR = r"C:\Users\ronin\PycharmProjects\PFinalproject"
CT_DIR   = os.path.join(BASE_DIR, "Cropped_Data", "All_Series_CT")
MASK_DIR = os.path.join(BASE_DIR, "Cropped_Data", "All_Series_Masks")
OUT_DIR  = os.path.join(BASE_DIR, "all_series_2d")
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_MAP = {
    "Patient_1_After":   ("Patient 1 - After Cancer",    1),
    "Patient_1_Before":  ("Patient 1 - Before (phases)", 1),
    "Patient_2_After":   ("Patient 2 - After Cancer",    1),
    "Patient_2_Before":  ("Patient 2 - Before Cancer",   1),
    "Patient_GA_Healthy":("Patient GA - Healthy",         0),
    "Patient_KB_Before": ("Patient KB - Before Cancer",   1),
    "Patient_KB_Cancer": ("Patient KB - Cancer",          1),
    "Patient_VT_Healthy":("Patient VT - Healthy",         0),
}

COLOR_CANCER  = "#FF4444"
COLOR_HEALTHY = "#44DD88"
COL_TITLES    = ["Axial (top-down)", "Coronal (front)", "Sagittal (side)"]


def window_hu(arr, center=60, width=350):
    lo, hi = center - width / 2, center + width / 2
    return np.clip(arr.astype(float), lo, hi)


# Group by patient
series_by_patient = defaultdict(list)
for fname in sorted(os.listdir(MASK_DIR)):
    if not fname.endswith(".nii.gz"):
        continue
    m = re.match(r"cropped_(Patient_\w+)_(s\d+)_mask\.nii\.gz", fname)
    if m:
        series_by_patient[m.group(1)].append((m.group(2), fname))

for patient_key, series_list in sorted(series_by_patient.items()):
    display, label = LABEL_MAP.get(patient_key, (patient_key, -1))
    n = len(series_list)
    print(f"\n{patient_key}: {n} series")

    lc = COLOR_CANCER if label == 1 else COLOR_HEALTHY
    label_str = "Cancer" if label == 1 else "Healthy"

    fig, axes = plt.subplots(n, 3, figsize=(13, n * 3.2))
    fig.patch.set_facecolor("#111111")
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"{display}  [{label_str}] — Cropped Liver: Axial / Coronal / Sagittal",
        fontsize=12, fontweight="bold", color=lc, y=1.01
    )
    for j, title in enumerate(COL_TITLES):
        axes[0, j].set_title(title, fontsize=10, color="white", pad=7, fontweight="bold")

    for row, (series_id, mask_fname) in enumerate(series_list):
        ct_fname  = mask_fname.replace("_mask.nii.gz", ".nii.gz")
        ct_path   = os.path.join(CT_DIR,   ct_fname)
        mask_path = os.path.join(MASK_DIR, mask_fname)

        if not os.path.exists(ct_path):
            for j in range(3):
                axes[row, j].set_facecolor("#222222")
                axes[row, j].text(0.5, 0.5, "CT not found", ha='center', va='center',
                                  transform=axes[row, j].transAxes, color='red')
                axes[row, j].axis('off')
            continue

        ct   = nib.load(ct_path).get_fdata()   # (X, Y, Z) in HU
        mask = nib.load(mask_path).get_fdata()
        X, Y, Z = ct.shape

        # Best slice: where liver mask is largest
        def best_ax(vol, axis):
            sums = vol.sum(axis=tuple(a for a in range(3) if a != axis))
            return int(np.argmax(sums))

        iz = best_ax(mask, 2)
        iy = best_ax(mask, 1)
        ix = best_ax(mask, 0)

        ct_ax = window_hu(ct[:, :, iz]).T
        mk_ax = mask[:, :, iz].T

        ct_co = window_hu(ct[:, iy, :]).T
        mk_co = mask[:, iy, :].T

        ct_sa = window_hu(ct[ix, :, :]).T
        mk_sa = mask[ix, :, :].T

        views = [(ct_ax, mk_ax), (ct_co, mk_co), (ct_sa, mk_sa)]

        for col, (ct_sl, mk_sl) in enumerate(views):
            ax = axes[row, col]
            ax.set_facecolor("#222222")
            ax.imshow(ct_sl, cmap="gray", origin="lower", aspect="auto")
            if mk_sl.sum() > 0:
                ov = np.ma.masked_where(mk_sl == 0, mk_sl.astype(float))
                ax.imshow(ov, cmap="autumn", alpha=0.60,
                          origin="lower", aspect="auto", vmin=0, vmax=1)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(lc)
                spine.set_linewidth(1.6)
            ax.axis("off")

        vol_vox = int(mask.sum())
        axes[row, 0].text(-0.12, 0.5, f"{series_id}\n{vol_vox:,} vox",
                          transform=axes[row, 0].transAxes,
                          va='center', ha='right', fontsize=8,
                          color=lc, fontweight='bold')
        print(f"  {series_id}: {vol_vox:,} liver voxels")

    patches = [
        mpatches.Patch(color="#FF6600", alpha=0.7, label="Liver mask"),
        mpatches.Patch(color=lc, label=f"{label_str} patient border"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.015),
               facecolor="#222222", edgecolor="#555555", labelcolor="white")

    plt.tight_layout(rect=[0.08, 0.02, 1, 1])
    out_path = os.path.join(OUT_DIR, f"{patient_key}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"  Saved: {out_path}")

print("\nDone. PNGs in all_series_2d/")
