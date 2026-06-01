"""
Analysis of Patient_VT misclassification:
  Part A — PCA + t-SNE: where does VT sit in feature space?
  Part B — Per-series breakdown: which specific series of VT look "cancer-like"?
"""

import sys, warnings, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler

from train_all_series_report import (
    clean_features, cv_filter, compute_icc,
    select_varfs, select_anova, augment, OUT_DIR,
)

CSV_PATH = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\all_series_radiomics.csv"

# ── Colours & markers per patient ────────────────────────────────────────────
PATIENT_STYLE = {
    "Patient_1":  {"color": "#E53935", "marker": "o",  "label": "Patient 1 (Cancer)"},
    "Patient_2":  {"color": "#FB8C00", "marker": "s",  "label": "Patient 2 (Cancer)"},
    "Patient_KB": {"color": "#C62828", "marker": "^",  "label": "Patient KB (Cancer)"},
    "Patient_GA": {"color": "#43A047", "marker": "D",  "label": "Patient GA (Healthy)"},
    "Patient_VT": {"color": "#1E88E5", "marker": "*",  "label": "Patient VT (Healthy)"},
}

# ── Load & preprocess ─────────────────────────────────────────────────────────
df        = pd.read_csv(CSV_PATH)
feat_cols = [c for c in df.columns if c not in ("series","patient","group","label")]
groups    = df["group"].tolist()
y_all     = df["label"].values.astype(int)
X_raw     = df[feat_cols].values.astype(float)

X_clean, fnames = clean_features(X_raw, list(feat_cols))
cv_keep         = cv_filter(X_clean, fnames)
icc_vals        = compute_icc(df, fnames)

idx_varfs, f_scores, _, _ = select_varfs(X_clean, y_all, fnames, icc_vals, cv_keep)
varfs_feats = [fnames[i] for i in idx_varfs]
X_top = X_clean[:, idx_varfs]   # only top-5 VaRFS features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_top)

print(f"Data: {len(df)} series, {len(varfs_feats)} features")
print(f"Features used: {varfs_feats}\n")

# ─────────────────────────────────────────────────────────────────────────────
# PART A — PCA + t-SNE
# ─────────────────────────────────────────────────────────────────────────────

def scatter_plot(ax, coords, df, title, explained=None):
    for _, row in df.iterrows():
        idx  = df.index.get_loc(row.name)
        grp  = row["group"]
        st   = PATIENT_STYLE[grp]
        size = 220 if grp == "Patient_VT" else 90
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=st["color"], marker=st["marker"],
                   s=size, edgecolors="white", linewidths=0.6, zorder=3)

    # Label VT series
    vt_mask = df["group"] == "Patient_VT"
    for i, row in df[vt_mask].iterrows():
        idx  = df.index.get_loc(i)
        name = row["series"].replace("Patient_VT_Healthy_", "VT_")
        ax.annotate(name, xy=(coords[idx, 0], coords[idx, 1]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=6.5, color="#1E88E5", fontweight="bold")

    # Cancer / Healthy boundary shading
    cancer_x  = coords[df["label"] == 1, 0]
    cancer_y  = coords[df["label"] == 1, 1]
    healthy_x = coords[df["label"] == 0, 0]
    healthy_y = coords[df["label"] == 0, 1]
    if len(cancer_x):
        ax.scatter([], [], c="#E53935", alpha=0.08, s=0)  # invisible — just for layout

    xlabel = (f"PC1 ({explained[0]:.1%} var)" if explained is not None else "Dim 1")
    ylabel = (f"PC2 ({explained[1]:.1%} var)" if explained is not None else "Dim 2")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)


# PCA
pca      = PCA(n_components=2, random_state=42)
X_pca    = pca.fit_transform(X_scaled)
exp_var  = pca.explained_variance_ratio_

# t-SNE (perplexity small because n=39)
tsne   = TSNE(n_components=2, perplexity=8, random_state=42,
              max_iter=2000, learning_rate="auto", init="pca")
X_tsne = tsne.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
df_reset = df.reset_index(drop=True)

scatter_plot(axes[0], X_pca,  df_reset, "PCA — Top-5 VaRFS Features", explained=exp_var)
scatter_plot(axes[1], X_tsne, df_reset, "t-SNE — Top-5 VaRFS Features")

# Legend
legend_handles = [
    Line2D([0],[0], marker=st["marker"], color="w", markerfacecolor=st["color"],
           markersize=9 if g=="Patient_VT" else 7, label=st["label"])
    for g, st in PATIENT_STYLE.items()
]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           fontsize=8.5, bbox_to_anchor=(0.5, -0.05))

fig.suptitle("Feature Space — Where Does Patient_VT Sit?",
             fontsize=14, fontweight="bold")
plt.tight_layout()
out_pca = os.path.join(OUT_DIR, "pca_tsne_feature_space.png")
plt.savefig(out_pca, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_pca}")

# PCA: print PC loadings so we understand WHICH features drive separation
print("\nPCA loadings (which features drive each component):")
for pc_idx in range(2):
    loadings = pca.components_[pc_idx]
    order    = np.argsort(np.abs(loadings))[::-1]
    print(f"  PC{pc_idx+1}: ", end="")
    parts = [f"{varfs_feats[i]}({loadings[i]:+.3f})" for i in order]
    print("  |  ".join(parts))

# ─────────────────────────────────────────────────────────────────────────────
# PART B — Per-series breakdown for Patient_VT vs Patient_GA
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("PER-SERIES BREAKDOWN: VT vs GA")
print("="*60)

# Train on everyone except VT, then predict each VT series individually
train_mask = np.array([g != "Patient_VT" for g in groups])
test_vt    = np.array([g == "Patient_VT"  for g in groups])
test_ga    = np.array([g == "Patient_GA"  for g in groups])

Xf_tr = X_clean[train_mask][:, idx_varfs]
y_tr  = y_all[train_mask]

sc = StandardScaler()
Xf_tr_s = sc.fit_transform(Xf_tr)
Xf_aug, y_aug = augment(Xf_tr_s, y_tr)

clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
clf.fit(Xf_aug, y_aug)

print("\nPatient_VT series (all should be Healthy = 0):")
print(f"  {'Series':<35} {'Prob(Cancer)':>13}  {'Decision':>10}  Feature values")
print(f"  {'-'*90}")

vt_probs = []
for i, row in df[test_vt].iterrows():
    x    = X_clean[i:i+1, idx_varfs]
    x_s  = sc.transform(x)
    prob = clf.predict_proba(x_s)[0, 1]
    pred = "Cancer" if prob >= 0.5 else "Healthy"
    flag = "<-- WRONG" if prob >= 0.5 else "OK"
    feat_vals = "  ".join([f"{f.split('_')[-1]}={x[0,j]:.1f}"
                           for j, f in enumerate(varfs_feats)])
    print(f"  {row['series']:<35} {prob:>13.3f}  {pred:>10}  {flag}")
    vt_probs.append((row["series"], prob))

print(f"\nPatient_GA series (all should be Healthy = 0):")
print(f"  {'Series':<35} {'Prob(Cancer)':>13}  {'Decision':>10}")
print(f"  {'-'*65}")
ga_probs = []
for i, row in df[test_ga].iterrows():
    x    = X_clean[i:i+1, idx_varfs]
    x_s  = sc.transform(x)
    prob = clf.predict_proba(x_s)[0, 1]
    pred = "Cancer" if prob >= 0.5 else "Healthy"
    flag = "<-- WRONG" if prob >= 0.5 else "OK"
    print(f"  {row['series']:<35} {prob:>13.3f}  {pred:>10}  {flag}")
    ga_probs.append((row["series"], prob))

# Feature value comparison: VT vs GA vs Cancer mean
print("\n" + "="*60)
print("FEATURE VALUES: VT vs GA vs Cancer mean")
print("="*60)
cancer_mask  = y_all == 1
healthy_mask = y_all == 0
print(f"\n  {'Feature':<35} {'Cancer mean':>12} {'GA mean':>10} {'VT mean':>10}  VT closer to:")
print(f"  {'-'*80}")
for j, feat in enumerate(varfs_feats):
    cancer_mean  = X_clean[cancer_mask,  idx_varfs[j]].mean()
    ga_mean      = X_clean[test_ga,      idx_varfs[j]].mean()
    vt_mean      = X_clean[test_vt,      idx_varfs[j]].mean()
    d_cancer = abs(vt_mean - cancer_mean)
    d_ga     = abs(vt_mean - ga_mean)
    closer   = "Cancer" if d_cancer < d_ga else "GA (Healthy)"
    arrow    = "  <--" if closer == "Cancer" else ""
    print(f"  {feat:<35} {cancer_mean:>12.3f} {ga_mean:>10.3f} {vt_mean:>10.3f}  {closer}{arrow}")

# ── Per-series probability bar chart ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

all_series  = [s for s, _ in vt_probs] + [s for s, _ in ga_probs]
all_probs   = [p for _, p in vt_probs] + [p for _, p in ga_probs]
all_colors  = ["#1E88E5"] * len(vt_probs) + ["#43A047"] * len(ga_probs)
short_names = [s.replace("Patient_VT_Healthy_","VT ").replace("Patient_GA_Healthy_","GA ")
               for s in all_series]

bars = ax.bar(short_names, all_probs, color=all_colors, edgecolor="white", linewidth=0.8)
ax.axhline(0.5, color="#E53935", linestyle="--", linewidth=1.5, label="Decision threshold (0.5)")
ax.axhline(0.7, color="#FF8F00", linestyle=":", linewidth=1.2, label="Threshold @ 0.7")

for bar, prob in zip(bars, all_probs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{prob:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_ylabel("P(Cancer)", fontsize=11)
ax.set_title("Per-Series Cancer Probability — Patient_VT (blue) vs Patient_GA (green)\n"
             "Trained on all other patients", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.1)
ax.tick_params(axis="x", rotation=35, labelsize=8)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines[["top","right"]].set_visible(False)

legend_patches = [
    mpatches.Patch(color="#1E88E5", label="Patient_VT (Healthy — misclassified)"),
    mpatches.Patch(color="#43A047", label="Patient_GA (Healthy — correct)"),
    Line2D([0],[0], color="#E53935", linestyle="--", label="Threshold 0.5"),
    Line2D([0],[0], color="#FF8F00", linestyle=":", label="Threshold 0.7"),
]
ax.legend(handles=legend_patches, fontsize=8.5, loc="upper right")

plt.tight_layout()
out_series = os.path.join(OUT_DIR, "per_series_vt_ga.png")
plt.savefig(out_series, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_series}")

# Threshold analysis
print("\n" + "="*60)
print("THRESHOLD ANALYSIS — what threshold fixes VT?")
print("="*60)
vt_max_prob = max(p for _, p in vt_probs)
cancer_probs = []
for i in range(len(df)):
    if y_all[i] == 1 and groups[i] != "Patient_VT":
        x   = X_clean[i:i+1, idx_varfs]
        x_s = sc.transform(x)
        cancer_probs.append(clf.predict_proba(x_s)[0, 1])

cancer_min = min(cancer_probs)
print(f"\n  Patient_VT max P(Cancer) across series: {vt_max_prob:.3f}")
print(f"  Cancer patients min P(Cancer):          {cancer_min:.3f}")
if vt_max_prob < cancer_min:
    print(f"\n  => A threshold between {vt_max_prob:.3f} and {cancer_min:.3f} would fix VT")
    print(f"     Recommended threshold: {(vt_max_prob + cancer_min)/2:.3f}")
else:
    print(f"\n  => Thresholds OVERLAP — no clean separation possible with current features")
    print(f"     VT max={vt_max_prob:.3f}  Cancer min={cancer_min:.3f}")

print(f"\nDone. Results in: {OUT_DIR}")
