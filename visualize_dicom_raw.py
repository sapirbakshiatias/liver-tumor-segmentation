"""
Shows raw DICOM images for Patient 1 Before and After.
Groups by SeriesInstanceUID, picks middle slice from each series,
displays as a grid PNG.
"""

import os
import numpy as np
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

BASE_DIR = r"C:\Users\ronin\PycharmProjects\PFinalproject"
OUT_DIR  = os.path.join(BASE_DIR, "dicom_raw_views")
os.makedirs(OUT_DIR, exist_ok=True)

PATIENTS = {
    "Patient_1_After":  os.path.join(BASE_DIR, "פציינט 1 אחרי סרטן",     "DICOM"),
    "Patient_1_Before": os.path.join(BASE_DIR, "מטופל 1 לפני 3 פאזות",   "DICOM"),
}

LABEL_MAP = {
    "Patient_1_After":  ("Patient 1 - After Cancer",    "#FF4444"),
    "Patient_1_Before": ("Patient 1 - Before (phases)", "#FF4444"),
}


def load_series(dicom_dir):
    """Returns dict: SeriesInstanceUID -> sorted list of (SliceLocation/InstanceNumber, pixel_array)"""
    series = defaultdict(list)
    for fname in os.listdir(dicom_dir):
        fpath = os.path.join(dicom_dir, fname)
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=False)
            uid  = str(getattr(ds, "SeriesInstanceUID",  "unknown"))
            loc  = float(getattr(ds, "SliceLocation",    getattr(ds, "InstanceNumber", 0)))
            desc = str(getattr(ds, "SeriesDescription",  ""))
            arr  = ds.pixel_array.astype(float)
            # Apply rescale if present
            slope = float(getattr(ds, "RescaleSlope",  1))
            inter = float(getattr(ds, "RescaleIntercept", 0))
            arr   = arr * slope + inter
            series[uid].append((loc, arr, desc))
        except Exception:
            continue

    # Sort each series by slice location
    for uid in series:
        series[uid].sort(key=lambda x: x[0])
    return series


def window(arr, center=60, width=350):
    lo, hi = center - width / 2, center + width / 2
    return np.clip(arr, lo, hi)


for patient_key, dicom_dir in PATIENTS.items():
    display, lc = LABEL_MAP[patient_key]
    print(f"\nLoading {patient_key} ...")

    series = load_series(dicom_dir)
    n = len(series)
    print(f"  Found {n} series")

    if n == 0:
        print("  No series found, skipping.")
        continue

    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4 + 0.8))
    fig.patch.set_facecolor("#111111")

    if nrows == 1 and ncols == 1:
        flat_axes = [axes]
    else:
        flat_axes = np.array(axes).flatten().tolist()

    fig.suptitle(f"{display} — Raw DICOM (middle slice per series)",
                 fontsize=12, fontweight="bold", color=lc, y=1.01)

    for i, (uid, slices) in enumerate(sorted(series.items())):
        ax = flat_axes[i]
        ax.set_facecolor("#222222")

        mid = len(slices) // 2
        _, arr, desc = slices[mid]
        arr_w = window(arr)

        ax.imshow(arr_w, cmap="gray", origin="upper", aspect="equal")
        short_uid = uid[-8:] if len(uid) > 8 else uid
        title = desc if desc else f"series …{short_uid}"
        ax.set_title(f"{title}\n({len(slices)} slices)", fontsize=7,
                     color="white", pad=3)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(lc)
            spine.set_linewidth(1.4)

        print(f"  Series '{desc}' ({len(slices)} slices)")

    for j in range(n, len(flat_axes)):
        flat_axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(OUT_DIR, f"{patient_key}_raw.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"  Saved: {out_path}")

print("\nDone. PNGs in dicom_raw_views/")
