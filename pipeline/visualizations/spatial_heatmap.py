"""
מפות חום מרחביות: גריד של פרוסות CT עם overlay של ערך פיצ'ר מקומי.
"""
import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

CT_DIR   = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\All_Series_CT"
MASK_DIR = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\All_Series_Masks"
OUT_DIR  = r"C:\Users\ronin\PycharmProjects\PFinalproject\results"
WINDOW, STRIDE = 16, 4
COLOR_CANCER  = "#C62828"
COLOR_HEALTHY = "#1B5E20"


def _load_slice(series_stem):
    ct_path   = os.path.join(CT_DIR,   f"cropped_{series_stem}.nii.gz")
    mask_path = os.path.join(MASK_DIR, f"cropped_{series_stem}_mask.nii.gz")
    if not os.path.exists(ct_path) or not os.path.exists(mask_path):
        return None, None
    ct   = nib.load(ct_path).get_fdata().astype(float)
    mask = nib.load(mask_path).get_fdata().astype(np.uint8)
    if ct.ndim != 3 or mask.ndim != 3:
        return None, None
    iz = int(np.argmax(mask.sum(axis=(0, 1))))
    return ct[:, :, iz].T, mask[:, :, iz].T


def _compute_patch_value(pix, feature_name):
    if "range" in feature_name:
        return float(pix.max() - pix.min())
    if "entropy" in feature_name:
        hist, _ = np.histogram(pix, bins=16, density=True)
        h_nz = hist[hist > 0]
        return float(-np.sum(h_nz * np.log2(h_nz + 1e-10)))
    if "p10" in feature_name:
        return float(np.percentile(pix, 10))
    if "homogeneity" in feature_name or "dissimilarity" in feature_name:
        if pix.max() == pix.min():
            return 1.0 if "homogeneity" in feature_name else 0.0
        norm  = np.clip(((pix - pix.min()) / (pix.max() - pix.min()) * 31), 0, 31).astype(np.uint8)
        g     = graycomatrix(norm.reshape(1, -1), [1], [0], levels=32, symmetric=True, normed=True)
        prop  = "homogeneity" if "homogeneity" in feature_name else "dissimilarity"
        return float(graycoprops(g, prop).mean())
    return float(pix.mean())


def _sliding_map(ct_sl, mask_sl, feature_name):
    h, w = ct_sl.shape
    out  = np.full((h, w), np.nan)
    for y0 in range(0, h - WINDOW + 1, STRIDE):
        for x0 in range(0, w - WINDOW + 1, STRIDE):
            pmask = mask_sl[y0:y0+WINDOW, x0:x0+WINDOW]
            if pmask.sum() < WINDOW // 2:
                continue
            pix = ct_sl[y0:y0+WINDOW, x0:x0+WINDOW][pmask > 0]
            val = _compute_patch_value(pix, feature_name)
            region = out[y0:y0+WINDOW, x0:x0+WINDOW]
            out[y0:y0+WINDOW, x0:x0+WINDOW] = np.where(np.isnan(region), val, region)
    return out


def plot_spatial_heatmaps(df, feat_names):
    series_list = df["series"].tolist()
    labels      = df["label"].tolist()
    n_cols = min(5, len(series_list))
    n_rows = (len(series_list) + n_cols - 1) // n_cols

    for feat in feat_names:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.2, n_rows*3.4))
        axes = np.array(axes).flatten()
        fig.suptitle(f"Spatial heatmap -- {feat}", fontsize=13, fontweight="bold", y=1.01)

        for idx, (stem, label) in enumerate(zip(series_list, labels)):
            ax = axes[idx]
            ct_sl, msk_sl = _load_slice(stem)
            if ct_sl is None:
                ax.axis("off"); continue

            ct_disp = np.clip(ct_sl, -100, 400)
            ct_disp = (ct_disp - ct_disp.min()) / (ct_disp.max() - ct_disp.min() + 1e-10)
            ax.imshow(ct_disp, cmap="gray", origin="lower", aspect="auto")

            fmap = _sliding_map(ct_sl, msk_sl, feat)
            im   = ax.imshow(np.ma.masked_where(msk_sl == 0, fmap),
                             cmap="hot", alpha=0.72, origin="lower", aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

            c = COLOR_CANCER if label == 1 else COLOR_HEALTHY
            ax.set_title(f"{stem}\n[{'Cancer' if label==1 else 'Healthy'}]",
                         fontsize=6.5, color=c, fontweight="bold", pad=3)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_edgecolor(c); spine.set_linewidth(2)

        for i in range(len(series_list), len(axes)):
            axes[i].axis("off")
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"spatial_{feat}.png")
        plt.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")
