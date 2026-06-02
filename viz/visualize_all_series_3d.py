"""
3D liver surface for ALL series per patient.
One PNG per patient saved to all_series_3d/
"""

import sys, os; sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")


import os
import re
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from collections import defaultdict

BASE_DIR     = r"C:\Users\ronin\PycharmProjects\PFinalproject"
CT_DIR       = os.path.join(BASE_DIR, "Cropped_Data", "All_Series_CT")
MASK_DIR     = os.path.join(BASE_DIR, "Cropped_Data", "All_Series_Masks")
OUT_DIR      = os.path.join(BASE_DIR, "all_series_3d")
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

COLOR_CANCER  = "#E53935"
COLOR_HEALTHY = "#43A047"

# Group cropped mask files by patient
series_by_patient = defaultdict(list)
for fname in sorted(os.listdir(MASK_DIR)):
    if not fname.endswith(".nii.gz"):
        continue
    # fname: cropped_Patient_1_After_s01_mask.nii.gz
    m = re.match(r"cropped_(Patient_\w+)_(s\d+)_mask\.nii\.gz", fname)
    if m:
        patient_key = m.group(1)
        series_id   = m.group(2)
        series_by_patient[patient_key].append((series_id, fname))

for patient_key, series_list in sorted(series_by_patient.items()):
    display, label = LABEL_MAP.get(patient_key, (patient_key, -1))
    n = len(series_list)
    print(f"\n{patient_key}: {n} series with liver")

    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(ncols * 5, nrows * 5))
    fig.patch.set_facecolor("#111111")
    color = COLOR_CANCER if label == 1 else COLOR_HEALTHY
    label_str = "Cancer" if label == 1 else "Healthy"
    fig.suptitle(f"{display}  [{label_str}] — 3D Liver per Series",
                 fontsize=13, fontweight="bold", color=color, y=1.01)

    for i, (series_id, mask_fname) in enumerate(series_list):
        mask_path = os.path.join(MASK_DIR, mask_fname)
        mask = nib.load(mask_path).get_fdata()

        try:
            verts, faces, _, _ = marching_cubes(mask, level=0.5, step_size=2)
        except Exception as e:
            print(f"  ERROR marching cubes {series_id}: {e}")
            continue

        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.set_facecolor("#1a1a1a")

        mesh = Poly3DCollection(verts[faces], alpha=0.80, linewidth=0)
        mesh.set_facecolor(color)
        mesh.set_edgecolor("none")
        ax.add_collection3d(mesh)

        ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
        ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
        ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())

        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#333333")
        ax.tick_params(colors="gray", labelsize=5)
        ax.set_xlabel("X", color="gray", fontsize=6)
        ax.set_ylabel("Y", color="gray", fontsize=6)
        ax.set_zlabel("Z", color="gray", fontsize=6)

        vol_vox = int(mask.sum())
        ax.set_title(f"{series_id}  |  {vol_vox:,} voxels",
                     fontsize=9, color="white", pad=3)
        ax.view_init(elev=30, azim=-60)
        print(f"  {series_id}: {vol_vox:,} voxels, mesh verts={len(verts)}")

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = os.path.join(OUT_DIR, f"{patient_key}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"  Saved: {out_path}")

print("\nDone. PNGs in all_series_3d/")
