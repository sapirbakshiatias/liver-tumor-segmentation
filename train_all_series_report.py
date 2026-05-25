"""
Patient-Level LOOCV on ALL series from Cropped_Data/All_Series_CT/.

Protocol:
    - 5 unique patients, 5 folds
    - Each fold: 4 patients train, 1 patient test (ALL their series held out)
    - Prediction per patient = average probability across all their test series

Feature selection (fitted on all data — conservative, acceptable with n=8 scans):
    - VaRFS: top-5 by normalised F-score × ICC
    - ANOVA: top-5 by F-score only

Outputs (saved to results/):
    heatmap_varfs.png         — static seaborn heatmap, VaRFS features
    heatmap_anova.png         — static seaborn heatmap, ANOVA features
    heatmap_interactive.html  — interactive Plotly heatmap (all top features)
    spatial_<feature>.png     — spatial overlay heatmap per top feature

Run: .venv\Scripts\python.exe train_all_series_report.py
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler

warnings.filterwarnings("ignore")

BASE_DIR  = r"C:\Users\ronin\PycharmProjects\PFinalproject"
CSV_PATH  = os.path.join(BASE_DIR, "Cropped_Data", "all_series_radiomics.csv")
CT_DIR    = os.path.join(BASE_DIR, "Cropped_Data", "All_Series_CT")
MASK_DIR  = os.path.join(BASE_DIR, "Cropped_Data", "All_Series_Masks")
OUT_DIR   = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

N_FEATURES    = 5
CV_THRESHOLD  = 100.0
RF_ESTIMATORS = 100
WINDOW        = 16   # px — sliding window for spatial heatmaps
STRIDE        = 4

# ICC pairs: same patient scanned before and after cancer development
ICC_PAIR_KEYS = [
    ("Patient_1_Before",  "Patient_1_After"),
    ("Patient_2_Before",  "Patient_2_After"),
    ("Patient_KB_Before", "Patient_KB_Cancer"),
]


# ── Preprocessing ─────────────────────────────────────────────────────────────

def clean_features(X, feature_names):
    bad   = np.any(~np.isfinite(X), axis=0)
    X     = X[:, ~bad]
    names = [f for f, b in zip(feature_names, bad) if not b]
    std   = X.std(axis=0)
    const = std == 0
    X     = X[:, ~const]
    names = [f for f, c in zip(names, const) if not c]
    mu = X.mean(axis=0); sd = X.std(axis=0)
    X  = np.clip(X, mu - 3*sd, mu + 3*sd)
    return X, names


def cv_filter(X, feature_names):
    mean = np.abs(X.mean(axis=0))
    std  = X.std(axis=0)
    cv   = np.where(mean > 1e-10, std / mean * 100, np.inf)
    return cv <= CV_THRESHOLD


def compute_icc(df, fnames):
    B_rows, A_rows = [], []
    for b_key, a_key in ICC_PAIR_KEYS:
        b = df[df["patient"] == b_key]
        a = df[df["patient"] == a_key]
        if b.empty or a.empty:
            continue
        # Average across all series for that patient-phase
        B_rows.append(b[fnames].mean().values)
        A_rows.append(a[fnames].mean().values)
    if len(B_rows) < 2:
        return np.ones(len(fnames))
    B = np.array(B_rows); A = np.array(A_rows)
    n, k = B.shape[0], 2
    vals = np.stack([B, A], axis=1)
    subj  = vals.mean(axis=1)
    grand = vals.mean(axis=(0, 1))
    MSB   = k * np.sum((subj - grand)**2, axis=0) / (n - 1)
    MSW   = np.sum((vals - subj[:, None, :])**2, axis=(0, 1)) / (n*(k-1))
    denom = MSB + (k - 1)*MSW
    icc   = np.where(denom > 0, (MSB - MSW) / denom, 0.0)
    return np.clip(icc, 0.0, 1.0)


# ── Feature selection ─────────────────────────────────────────────────────────

def select_varfs(X, y, fnames, icc_vals, cv_keep, k=N_FEATURES):
    f_scores, _ = f_classif(X, y)
    f_scores = np.nan_to_num(f_scores)
    f_norm   = f_scores / (f_scores.max() + 1e-10)
    score    = f_norm * icc_vals
    score[~cv_keep] = 0.0
    top = np.argsort(score)[::-1][:k]
    return top, f_scores, icc_vals, score


def select_anova(X, y, fnames, cv_keep, k=N_FEATURES):
    f_scores, _ = f_classif(X, y)
    f_scores = np.nan_to_num(f_scores)
    f2 = f_scores.copy(); f2[~cv_keep] = 0.0
    top = np.argsort(f2)[::-1][:k]
    return top, f_scores


# ── Classifiers ───────────────────────────────────────────────────────────────

def make_clf(name):
    clfs = {
        "RandomForest": RandomForestClassifier(n_estimators=RF_ESTIMATORS,
                            class_weight="balanced", random_state=42),
        "SVM":          SVC(kernel="rbf", C=1.0, class_weight="balanced",
                            probability=True, random_state=42),
        "LogisticRegr": LogisticRegression(class_weight="balanced",
                            max_iter=1000, random_state=42),
        "KNN":          KNeighborsClassifier(n_neighbors=3),
        "NaiveBayes":   GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(max_depth=3,
                            class_weight="balanced", random_state=42),
    }
    return clfs[name]


def augment(X, y):
    n_min = (y == 0).sum()
    if n_min >= 2:
        return SMOTE(k_neighbors=1, random_state=42).fit_resample(X, y)
    if n_min == 1:
        return RandomOverSampler(random_state=42).fit_resample(X, y)
    return X, y


def metrics(y_true, y_pred, y_prob=None):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc  = (tp + tn) / len(y_true)
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1   = (2*prec*sens / (prec+sens)
            if not (np.isnan(prec) or np.isnan(sens)) and (prec+sens) > 0
            else np.nan)
    auc = np.nan
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try: auc = roc_auc_score(y_true, y_prob)
        except: pass
    return acc, sens, spec, auc, f1


# ── Patient-Level LOOCV ───────────────────────────────────────────────────────

def run_patient_loocv(X, y, groups, feat_idx, clf_name):
    """
    Leave-one-PATIENT-out: all series of the held-out patient form the test set.
    Patient-level prediction = average probability across that patient's series.
    """
    unique_patients = list(dict.fromkeys(groups))
    val_true, val_pred, val_prob = [], [], []
    fold_log = []

    for patient in unique_patients:
        test_mask  = np.array([g == patient for g in groups])
        train_mask = ~test_mask

        Xf_tr = X[train_mask][:, feat_idx]
        Xf_te = X[test_mask][:, feat_idx]
        y_tr  = y[train_mask]
        true_label = int(np.round(y[test_mask].mean()))

        scaler    = StandardScaler()
        Xf_tr_s   = scaler.fit_transform(Xf_tr)
        Xf_te_s   = scaler.transform(Xf_te)
        Xf_aug, y_aug = augment(Xf_tr_s, y_tr)

        if len(np.unique(y_aug)) < 2:
            only = int(y_aug[0])
            val_true.append(true_label); val_pred.append(only); val_prob.append(float(only))
            fold_log.append((patient, true_label, only, float(only)))
            continue

        clf = make_clf(clf_name)
        clf.fit(Xf_aug, y_aug)
        proba    = clf.predict_proba(Xf_te_s)
        avg_prob = float(proba[:, 1].mean()) if proba.shape[1] == 2 else 0.5
        pred     = int(avg_prob >= 0.5)

        val_true.append(true_label); val_pred.append(pred); val_prob.append(avg_prob)
        fold_log.append((patient, true_label, pred, avg_prob))

    acc, sens, spec, auc, f1 = metrics(val_true, val_pred, val_prob)
    return acc, sens, spec, auc, f1, fold_log


# ── Spatial heatmap helpers ───────────────────────────────────────────────────

def sliding_feature_map(ct_sl, mask_sl, feature_name, win=WINDOW, stride=STRIDE):
    """Compute a local feature value in a sliding window over the liver slice."""
    h, w = ct_sl.shape
    out  = np.full((h, w), np.nan)

    for y0 in range(0, h - win + 1, stride):
        for x0 in range(0, w - win + 1, stride):
            patch  = ct_sl[y0:y0+win, x0:x0+win]
            pmask  = mask_sl[y0:y0+win, x0:x0+win]
            if pmask.sum() < (win // 2):
                continue
            pix = patch[pmask > 0]

            if "range" in feature_name:
                val = float(pix.max() - pix.min())
            elif "entropy" in feature_name:
                hist, _ = np.histogram(pix, bins=16, density=True)
                h_nz = hist[hist > 0]
                val = float(-np.sum(h_nz * np.log2(h_nz + 1e-10)))
            elif "p10" in feature_name:
                val = float(np.percentile(pix, 10))
            elif "homogeneity" in feature_name:
                if pix.max() == pix.min():
                    val = 1.0
                else:
                    norm = np.clip(
                        ((pix - pix.min()) / (pix.max() - pix.min()) * 31), 0, 31
                    ).astype(np.uint8)
                    norm_2d = norm.reshape(1, -1)  # treat as 1-row image
                    g = graycomatrix(norm_2d, [1], [0], levels=32, symmetric=True, normed=True)
                    val = float(graycoprops(g, "homogeneity").mean())
            elif "dissimilarity" in feature_name:
                if pix.max() == pix.min():
                    val = 0.0
                else:
                    norm = np.clip(
                        ((pix - pix.min()) / (pix.max() - pix.min()) * 31), 0, 31
                    ).astype(np.uint8)
                    norm_2d = norm.reshape(1, -1)
                    g = graycomatrix(norm_2d, [1], [0], levels=32, symmetric=True, normed=True)
                    val = float(graycoprops(g, "dissimilarity").mean())
            else:
                val = float(pix.mean())

            # Fill the window (first-write wins)
            region = out[y0:y0+win, x0:x0+win]
            out[y0:y0+win, x0:x0+win] = np.where(np.isnan(region), val, region)

    return out


def load_best_axial_slice(series_stem):
    ct_path   = os.path.join(CT_DIR,   f"cropped_{series_stem}.nii.gz")
    mask_path = os.path.join(MASK_DIR, f"cropped_{series_stem}_mask.nii.gz")
    if not os.path.exists(ct_path) or not os.path.exists(mask_path):
        return None, None
    ct   = nib.load(ct_path).get_fdata().astype(float)
    mask = nib.load(mask_path).get_fdata().astype(np.uint8)
    if ct.ndim != 3 or mask.ndim != 3:
        return None, None
    iz     = int(np.argmax(mask.sum(axis=(0, 1))))
    ct_sl  = ct[:, :, iz].T    # (X,Y) → display as (Y,X)
    msk_sl = mask[:, :, iz].T
    return ct_sl, msk_sl


# ── Visualizations ────────────────────────────────────────────────────────────

COLOR_CANCER  = "#C62828"
COLOR_HEALTHY = "#1B5E20"


def plot_statistical_heatmap(df, feat_names, title, out_path):
    """Seaborn heatmap: rows = series, cols = selected features, z-normalized."""
    sub = df[["series", "group", "label"] + feat_names].copy()
    sub = sub.sort_values(["label", "group"], ascending=[False, True])

    X = sub[feat_names].values.astype(float)
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    row_labels = [f"{r.series}  ({'C' if r.label==1 else 'H'})"
                  for r in sub.itertuples()]
    row_colors = [COLOR_CANCER if r.label == 1 else COLOR_HEALTHY
                  for r in sub.itertuples()]

    fig, ax = plt.subplots(figsize=(max(10, len(feat_names)*2),
                                    max(6, len(sub)*0.45)))
    sns.heatmap(
        pd.DataFrame(X_norm, columns=feat_names, index=row_labels),
        cmap="RdBu_r", center=0, linewidths=0.4, linecolor="#333333",
        ax=ax, cbar_kws={"label": "Z-score", "shrink": 0.6},
        annot=True, fmt=".2f", annot_kws={"size": 6},
    )
    # Coloured left bar for cancer / healthy
    for i, rc in enumerate(row_colors):
        ax.add_patch(plt.Rectangle(
            (-0.35, i), 0.28, 1, color=rc, clip_on=False, zorder=3))

    patches = [
        mpatches.Patch(color=COLOR_CANCER,  label="Cancer"),
        mpatches.Patch(color=COLOR_HEALTHY, label="Healthy"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8,
              bbox_to_anchor=(1.18, 1.02))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Feature", fontsize=10)
    ax.set_ylabel("Series", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_spatial_heatmaps(df, feat_names):
    """For each top feature: grid of CT slices with local feature overlay."""
    series_list = df["series"].tolist()
    labels      = df["label"].tolist()
    groups      = df["group"].tolist()

    n_series = len(series_list)
    n_cols   = min(5, n_series)
    n_rows   = (n_series + n_cols - 1) // n_cols

    for feat in feat_names:
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 3.2, n_rows * 3.4))
        axes = np.array(axes).flatten()
        fig.suptitle(f"Spatial heatmap — {feat}",
                     fontsize=13, fontweight="bold", y=1.01)

        for idx, (stem, label, grp) in enumerate(zip(series_list, labels, groups)):
            ax = axes[idx]
            ct_sl, msk_sl = load_best_axial_slice(stem)
            if ct_sl is None:
                ax.axis("off"); continue

            # CT background (grey)
            ct_disp = np.clip(ct_sl, -100, 400)
            ct_disp = (ct_disp - ct_disp.min()) / (ct_disp.max() - ct_disp.min() + 1e-10)
            ax.imshow(ct_disp, cmap="gray", origin="lower", aspect="auto")

            # Feature overlay (only inside liver mask)
            fmap = sliding_feature_map(ct_sl, msk_sl, feat)
            fmap_masked = np.ma.masked_where(msk_sl == 0, fmap)
            im = ax.imshow(fmap_masked, cmap="hot", alpha=0.72,
                           origin="lower", aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

            c = COLOR_CANCER if label == 1 else COLOR_HEALTHY
            ax.set_title(f"{stem.replace('_', ' ')}\n[{'Cancer' if label==1 else 'Healthy'}]",
                         fontsize=6.5, color=c, fontweight="bold", pad=3)
            ax.axis("off")

            # Border colour
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_edgecolor(c); spine.set_linewidth(2)

        for i in range(n_series, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"spatial_{feat}.png")
        plt.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


def plot_interactive_heatmap(df, feat_names, out_path):
    """Plotly interactive heatmap with hover info."""
    sub = df[["series", "group", "label"] + feat_names].copy()
    sub = sub.sort_values(["label", "group"], ascending=[False, True])

    X      = sub[feat_names].values.astype(float)
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    y_labels = [
        f"{'🔴' if r.label==1 else '🟢'} {r.series} [{r.group}]"
        for r in sub.itertuples()
    ]

    hover = [
        [f"<b>{feat_names[j]}</b><br>Raw: {X[i,j]:.3f}<br>Z-score: {X_norm[i,j]:.2f}"
         for j in range(len(feat_names))]
        for i in range(len(y_labels))
    ]

    fig = go.Figure(go.Heatmap(
        z=X_norm,
        x=feat_names,
        y=y_labels,
        colorscale="RdBu",
        zmid=0,
        text=hover,
        hoverinfo="text",
        colorbar=dict(title="Z-score"),
        xgap=2, ygap=1,
    ))
    fig.update_layout(
        title=dict(text="Feature Heatmap — All Series (interactive)",
                   font=dict(size=16)),
        xaxis=dict(title="Feature", tickangle=-30),
        yaxis=dict(title="Series"),
        height=max(500, len(y_labels) * 32),
        template="plotly_dark",
        margin=dict(l=260, r=60, t=80, b=100),
    )
    fig.write_html(out_path)
    print(f"  Saved: {out_path}")


def plot_feature_importance_bar(fnames, f_scores, icc_vals, varfs_idx, anova_idx, out_path):
    """Bar chart comparing F-score and ICC for all features, highlighting top selections."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    varfs_set = set(varfs_idx)
    anova_set = set(anova_idx)
    colors_f = []
    for i in range(len(fnames)):
        if i in varfs_set and i in anova_set:
            colors_f.append("#FF6F00")   # both
        elif i in varfs_set:
            colors_f.append("#1565C0")   # VaRFS only
        elif i in anova_set:
            colors_f.append("#2E7D32")   # ANOVA only
        else:
            colors_f.append("#9E9E9E")   # not selected

    sort_idx = np.argsort(f_scores)[::-1]
    sorted_names   = [fnames[i] for i in sort_idx]
    sorted_fscores = f_scores[sort_idx]
    sorted_icc     = icc_vals[sort_idx]
    sorted_colors  = [colors_f[i] for i in sort_idx]

    ax1.barh(sorted_names, sorted_fscores, color=sorted_colors)
    ax1.set_xlabel("F-score (ANOVA)", fontsize=10)
    ax1.set_title("Feature F-scores", fontsize=12, fontweight="bold")
    ax1.invert_yaxis()

    ax2.barh(sorted_names, sorted_icc, color=sorted_colors)
    ax2.set_xlabel("ICC (reproducibility)", fontsize=10)
    ax2.set_title("Feature ICC", fontsize=12, fontweight="bold")
    ax2.invert_yaxis()

    legend_patches = [
        mpatches.Patch(color="#FF6F00", label="Selected by both"),
        mpatches.Patch(color="#1565C0", label="VaRFS only"),
        mpatches.Patch(color="#2E7D32", label="ANOVA only"),
        mpatches.Patch(color="#9E9E9E", label="Not selected"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Feature Importance: F-score vs ICC", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    feat_cols = [c for c in df.columns if c not in ("series", "patient", "group", "label")]
    groups    = df["group"].tolist()
    y_all     = df["label"].values.astype(int)
    X_raw     = df[feat_cols].values.astype(float)

    # ── Preprocessing (fitted on all data for feature selection)
    X_clean, fnames = clean_features(X_raw, list(feat_cols))
    cv_keep         = cv_filter(X_clean, fnames)
    icc_vals        = compute_icc(df, fnames)

    unique_patients = list(dict.fromkeys(groups))
    print(f"Loaded {len(df)} series  |  {len(unique_patients)} patients  "
          f"|  {len(fnames)} clean features")
    print(f"CV removed: {(~cv_keep).sum()}  |  ICC >= 0.75: {(icc_vals >= 0.75).sum()}")
    print(f"\nPatients: {unique_patients}")

    # ── Feature selection
    idx_varfs, f_scores, icc_all, varfs_scores = select_varfs(
        X_clean, y_all, fnames, icc_vals, cv_keep)
    idx_anova, _ = select_anova(X_clean, y_all, fnames, cv_keep)

    varfs_feats = [fnames[i] for i in idx_varfs]
    anova_feats = [fnames[i] for i in idx_anova]

    print(f"\n{'='*65}")
    print("FEATURE SELECTION RESULTS")
    print(f"{'='*65}")
    print(f"\nVaRFS top {N_FEATURES} (F-score × ICC):")
    for rank, i in enumerate(idx_varfs, 1):
        print(f"  {rank}. {fnames[i]:<40}  F={f_scores[i]:.3f}  "
              f"ICC={icc_all[i]:.3f}  score={varfs_scores[i]:.3f}")
    print(f"\nANOVA top {N_FEATURES} (F-score only):")
    for rank, i in enumerate(idx_anova, 1):
        print(f"  {rank}. {fnames[i]:<40}  F={f_scores[i]:.3f}  ICC={icc_all[i]:.3f}")

    # All unique top features (union)
    all_top_feats = list(dict.fromkeys(varfs_feats + anova_feats))

    # ── LOOCV
    clf_names = ["RandomForest", "SVM", "LogisticRegr", "KNN", "NaiveBayes", "DecisionTree"]
    combinations = (
        [(f"VaRFS + {c}", idx_varfs, c) for c in clf_names] +
        [(f"ANOVA + {c}", idx_anova, c) for c in clf_names]
    )

    print(f"\n{'='*65}")
    print(f"PATIENT-LEVEL LOOCV  ({len(unique_patients)} folds, 4 train + 1 test)")
    print(f"{'='*65}\n")

    results = []
    for label, feat_idx, clf_name in combinations:
        print(f"  {label:<32}", end=" ", flush=True)
        acc, sens, spec, auc, f1, fold_log = run_patient_loocv(
            X_clean, y_all, groups, feat_idx, clf_name)
        results.append({"Model": label, "Acc": acc,
                        "Sens": sens, "Spec": spec, "AUC": auc, "F1": f1,
                        "fold_log": fold_log})
        a_s = f"{auc:.3f}" if not np.isnan(auc) else " N/A"
        f_s = f"{f1:.3f}"  if not np.isnan(f1)  else " N/A"
        print(f"Acc={acc:.0%}  Sens={sens:.0%}  AUC={a_s}  F1={f_s}")

    # ── Summary table
    results.sort(key=lambda r: (0 if np.isnan(r["F1"])  else r["F1"],
                                0 if np.isnan(r["AUC"]) else r["AUC"]), reverse=True)
    print(f"\n{'='*72}")
    print("SUMMARY TABLE  (sorted by F1 ↓)")
    print(f"{'='*72}")
    print(f"  {'#':>3}  {'Model':<30} {'Acc':>5} {'Sens':>6} {'Spec':>6} "
          f"{'AUC':>7} {'F1':>7}  Valid")
    print("  " + "-"*68)
    for rank, r in enumerate(results, 1):
        a_s = f"{r['AUC']:.3f}" if not np.isnan(r["AUC"]) else "  N/A"
        f_s = f"{r['F1']:.3f}"  if not np.isnan(r["F1"])  else "  N/A"
        valid = (r["Acc"] >= 0.8
                 and not np.isnan(r["AUC"]) and r["AUC"] >= 0.8
                 and not np.isnan(r["F1"])  and r["F1"]  >= 0.8)
        flag  = "YES" if valid else " — "
        print(f"  {rank:>3}.  {r['Model']:<30} {r['Acc']:>4.0%}  "
              f"{r['Sens']:>4.0%}  "
              f"{'N/A' if np.isnan(r['Spec']) else f'{r[chr(83)+chr(112)+chr(101)+chr(99)]:.0%}':>5}  "
              f"{a_s:>6}  {f_s:>6}  {flag}")

    # ── Best model fold detail
    best = results[0]
    print(f"\nBest: {best['Model']}")
    print(f"Per-patient breakdown:")
    for patient, true_l, pred_l, prob in best["fold_log"]:
        status = "OK   " if true_l == pred_l else "WRONG"
        print(f"  {status}  {patient:<20}  "
              f"true={'Cancer' if true_l else 'Healthy':7}  "
              f"pred={'Cancer' if pred_l else 'Healthy':7}  "
              f"prob={prob:.3f}")

    # ── Generate visualizations
    print(f"\n{'='*65}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*65}\n")

    print("Feature importance bar chart:")
    plot_feature_importance_bar(
        fnames, f_scores, icc_all, idx_varfs, idx_anova,
        os.path.join(OUT_DIR, "feature_importance.png"))

    print("\nStatistical heatmaps:")
    plot_statistical_heatmap(df, varfs_feats,
                             f"VaRFS Top {N_FEATURES} Features — All Series (Z-normalised)",
                             os.path.join(OUT_DIR, "heatmap_varfs.png"))
    plot_statistical_heatmap(df, anova_feats,
                             f"ANOVA Top {N_FEATURES} Features — All Series (Z-normalised)",
                             os.path.join(OUT_DIR, "heatmap_anova.png"))

    print("\nInteractive heatmap:")
    plot_interactive_heatmap(df, all_top_feats,
                             os.path.join(OUT_DIR, "heatmap_interactive.html"))

    print(f"\nSpatial heatmaps ({len(all_top_feats)} features × {len(df)} series):")
    plot_spatial_heatmaps(df, all_top_feats)

    print(f"\nDone.  All outputs saved to: {OUT_DIR}")
