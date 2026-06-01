"""
Animated GIF through all liver slices for each patient.
Shows all series side by side, animated from top to bottom of the cropped liver.
Output: all_series_gif/<patient>.gif
"""

import os
import re
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict

BASE_DIR = r"C:\Users\ronin\PycharmProjects\PFinalproject"
CT_DIR   = os.path.join(BASE_DIR, "Cropped_Data", "All_Series_CT")
MASK_DIR = os.path.join(BASE_DIR, "Cropped_Data", "All_Series_Masks")
OUT_DIR  = os.path.join(BASE_DIR, "all_series_gif")
os.makedirs(OUT_DIR, exist_ok=True)

N_FRAMES = 50   # frames per GIF
FPS      = 12   # frames per second

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

HU_MIN, HU_MAX = -100, 400


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
    lc = COLOR_CANCER if label == 1 else COLOR_HEALTHY
    label_str = "Cancer" if label == 1 else "Healthy"

    # Load all volumes for this patient
    volumes = []
    for series_id, mask_fname in series_list:
        ct_fname  = mask_fname.replace("_mask.nii.gz", ".nii.gz")
        ct_path   = os.path.join(CT_DIR,   ct_fname)
        mask_path = os.path.join(MASK_DIR, mask_fname)
        if not os.path.exists(ct_path):
            continue
        ct   = nib.load(ct_path).get_fdata()    # (X, Y, Z) in HU
        mask = nib.load(mask_path).get_fdata()
        volumes.append((series_id, ct, mask))

    if not volumes:
        continue

    n = len(volumes)
    print(f"\n{patient_key}: {n} series, building GIF...")

    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 4.2 + 0.7))
    fig.patch.set_facecolor("#111111")

    # Flatten axes
    if nrows == 1 and ncols == 1:
        flat_axes = [axes]
    elif nrows == 1 or ncols == 1:
        flat_axes = list(np.array(axes).flatten())
    else:
        flat_axes = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    # Title (will be updated each frame)
    title = fig.suptitle(
        f"{display}  [{label_str}]  —  top of liver",
        fontsize=11, fontweight="bold", color=lc, y=0.99
    )

    # Create initial plots
    im_list = []
    ov_list = []

    for i, (series_id, ct, mask) in enumerate(volumes):
        ax = flat_axes[i]
        ax.set_facecolor("#222222")

        z0 = 0
        ct_sl = window_hu(ct[:, :, z0]).T
        mk_sl = mask[:, :, z0].T

        im = ax.imshow(ct_sl, cmap="gray", origin="lower", aspect="auto",
                       vmin=HU_MIN, vmax=HU_MAX, animated=True)
        mk_m = np.ma.masked_where(mk_sl == 0, np.ones_like(mk_sl, dtype=float))
        ov = ax.imshow(mk_m, cmap="autumn", alpha=0.60,
                       origin="lower", aspect="auto", vmin=0, vmax=1, animated=True)

        vol_vox = int(mask.sum())
        ax.set_title(f"{series_id}  ({vol_vox:,} vox)", fontsize=8,
                     color="white", pad=3)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(lc)
            spine.set_linewidth(1.4)

        im_list.append(im)
        ov_list.append(ov)

    # Hide unused subplots
    for i in range(len(volumes), len(flat_axes)):
        flat_axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    def update(frame):
        pct = frame / (N_FRAMES - 1)
        artists = []
        for i, (series_id, ct, mask) in enumerate(volumes):
            Z  = ct.shape[2]
            iz = int(pct * (Z - 1))

            ct_sl = window_hu(ct[:, :, iz]).T
            im_list[i].set_data(ct_sl)

            mk_sl = mask[:, :, iz].T
            mk_m  = np.ma.masked_where(mk_sl == 0, np.ones_like(mk_sl, dtype=float))
            ov_list[i].set_data(mk_m)

            artists.extend([im_list[i], ov_list[i]])

        depth_pct = int(pct * 100)
        if depth_pct < 15:
            depth_label = "top of liver"
        elif depth_pct > 85:
            depth_label = "bottom of liver"
        else:
            depth_label = f"{depth_pct}% depth"
        title.set_text(f"{display}  [{label_str}]  —  {depth_label}")
        artists.append(title)
        return artists

    ani = animation.FuncAnimation(
        fig, update, frames=N_FRAMES, interval=1000 // FPS, blit=False
    )

    out_path = os.path.join(OUT_DIR, f"{patient_key}.gif")
    ani.save(out_path, writer="pillow", fps=FPS, dpi=110)
    plt.close(fig)
    print(f"  Saved: {out_path}")

print(f"\nDone. GIFs in all_series_gif/")
