"""
Patient-level LOOCV using combined 2D slice + 3D series features.

Each slice gets 54 features:
  - 16 axial 2D features  (slice-specific texture)
  - 38 3D volume features  (sagittal/coronal GLCM, shape, gradient from whole liver)

This gives every axial slice access to sagittal_glcm_dissimilarity —
the key VaRFS feature that the pure 2D approach cannot compute.

Patient prediction = weighted average of per-slice probabilities
                     (weight = liver_px / max_liver_px_of_patient)

Run: .venv\Scripts\python.exe training\train_2d_plus_3d.py
"""
import sys, os, warnings
sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler

from pipeline.clean_features  import clean_features, cv_filter
from pipeline.compute_icc     import compute_icc
from pipeline.select_features import select_varfs, select_anova
from pipeline.metrics         import metrics

CSV_PATH = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\slice_2d_plus_3d.csv"

# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)

METADATA = {"series", "patient", "group", "label", "slice_idx", "liver_px"}
feat_cols = [c for c in df.columns if c not in METADATA]

# Drop rows with NaN or Inf in features
df = df.dropna(subset=feat_cols)
df = df[np.all(np.isfinite(df[feat_cols].values), axis=1)]

groups   = df["group"].tolist()
y_all    = df["label"].values.astype(int)
X_raw    = df[feat_cols].values.astype(float)
PATIENTS = list(dict.fromkeys(groups))

print(f"Dataset: {len(df)} slices | {len(PATIENTS)} patients | {len(feat_cols)} raw features")

# ── Feature cleaning ──────────────────────────────────────────────────────────

X_clean, fnames = clean_features(X_raw, list(feat_cols))
cv_keep         = cv_filter(X_clean, fnames)
icc_vals        = compute_icc(df, fnames)

idx_varfs, f_scores, _, varfs_scores = select_varfs(X_clean, y_all, fnames, icc_vals, cv_keep)
idx_anova, _                          = select_anova(X_clean, y_all, fnames, cv_keep)

print(f"After cleaning: {len(fnames)} features")
print(f"\nVaRFS top-5:")
for i, idx in enumerate(idx_varfs):
    print(f"  {i+1}. {fnames[idx]:<45}  F={f_scores[idx]:.2f}  ICC={icc_vals[idx]:.3f}")
print(f"\nANOVA top-5:")
for i, idx in enumerate(idx_anova):
    print(f"  {i+1}. {fnames[idx]:<45}  F={f_scores[idx]:.2f}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def augment(X, y):
    n_min = (y == 0).sum()
    if n_min >= 2:
        return SMOTE(k_neighbors=min(1, n_min-1), random_state=42).fit_resample(X, y)
    if n_min == 1:
        return RandomOverSampler(random_state=42).fit_resample(X, y)
    return X, y


def make_clf(name):
    return {
        "KNN":           KNeighborsClassifier(n_neighbors=5),
        "RandomForest":  RandomForestClassifier(n_estimators=200,
                             class_weight="balanced", random_state=42),
        "GradientBoost": GradientBoostingClassifier(n_estimators=100,
                             max_depth=3, learning_rate=0.1, random_state=42),
        "SVM":           SVC(kernel="rbf", C=1.0, class_weight="balanced",
                             probability=True, random_state=42),
        "MLP":           MLPClassifier(hidden_layer_sizes=(64, 32),
                             max_iter=500, random_state=42,
                             early_stopping=True, validation_fraction=0.1),
    }[name]


def loocv(feat_idx, clf_name):
    """Patient-level LOOCV with liver_px weighted slice aggregation."""
    fold_true, fold_pred, fold_prob = [], [], []
    per_patient = {}

    for patient in PATIENTS:
        test_mask  = np.array([g == patient for g in groups])
        train_mask = ~test_mask

        X_tr = X_clean[train_mask][:, feat_idx]
        X_te = X_clean[test_mask][:, feat_idx]
        y_tr = y_all[train_mask]
        true_label = int(y_all[test_mask][0])

        sc      = StandardScaler()
        X_tr_s  = sc.fit_transform(X_tr)
        X_te_s  = sc.transform(X_te)
        X_aug, y_aug = augment(X_tr_s, y_tr)

        clf = make_clf(clf_name)
        clf.fit(X_aug, y_aug)
        slice_probs = clf.predict_proba(X_te_s)[:, 1]

        # Weight each slice by its liver pixel count (normalized per patient)
        lx      = df[test_mask]["liver_px"].values.astype(float)
        weights = lx / lx.max()
        avg_prob = float(np.average(slice_probs, weights=weights))
        pred     = int(avg_prob >= 0.5)

        fold_true.append(true_label)
        fold_pred.append(pred)
        fold_prob.append(avg_prob)
        per_patient[patient] = (true_label, pred, avg_prob)

    acc, sens, spec, auc, f1 = metrics(fold_true, fold_pred, fold_prob)
    return acc, sens, spec, auc, f1, per_patient


# ── Run all models ────────────────────────────────────────────────────────────

CLF_NAMES = ["KNN", "RandomForest", "GradientBoost", "SVM", "MLP"]

SEP = "=" * 130

print(f"\n{SEP}")
print("2D+3D COMBINED FEATURES — Patient-Level LOOCV")
print("Each slice: 16 axial features + 38 volume features from parent series")
print(f"{SEP}")

def cell(pp, p):
    tl, pl, prob = pp[p]
    status = "OK" if tl == pl else ("FP" if pl==1 and tl==0 else "FN")
    return f"{status}({prob:.2f})"

header = (f"  {'Combo':<28} {'Acc':>5} {'Sens':>5} {'Spec':>5} {'AUC':>6} {'F1':>6} | "
          f"{'P1':>10} {'P2':>10} {'KB':>10} {'GA':>10} {'VT':>10}")
print(header)
print("  " + "-"*128)

all_results = []

for feat_label, feat_idx in [("VaRFS+", idx_varfs), ("ANOVA+", idx_anova)]:
    for clf_name in CLF_NAMES:
        acc, sens, spec, auc, f1, pp = loocv(feat_idx, clf_name)
        auc_s = f"{auc:.3f}" if auc==auc else "  N/A"
        f1_s  = f"{f1:.3f}"  if f1==f1   else "  N/A"
        wrong = [p for p in PATIENTS if pp[p][0] != pp[p][1]]
        errs  = "  <- " + ", ".join(
            f"{p}({'FP' if pp[p][1]==1 and pp[p][0]==0 else 'FN'})" for p in wrong
        ) if wrong else ""

        row = f"  {feat_label+clf_name:<28} {acc:>4.0%} {sens:>5.0%} {spec:>5.0%} {auc_s:>6} {f1_s:>6} | "
        row += f"{cell(pp,'Patient_1'):>10} {cell(pp,'Patient_2'):>10} {cell(pp,'Patient_KB'):>10} "
        row += f"{cell(pp,'Patient_GA'):>10} {cell(pp,'Patient_VT'):>10}{errs}"
        print(row)
        all_results.append((feat_label+clf_name, acc, sens, spec, auc, f1, pp))

# ── Comparison table ──────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("COMPARISON: 2D-only vs 2D+3D vs 3D-only (VaRFS+KNN reference)")
print(f"{SEP}")
print(f"  {'Approach':<35} {'Acc':>5} {'Sens':>5} {'Spec':>5} {'AUC':>6} {'F1':>6}")
print("  " + "-"*65)

# Reference results (hardcoded from previous runs)
refs = [
    ("3D VaRFS+KNN (reference)",          1.00, 1.00, 1.00, 1.000, 1.000),
    ("2D-only RF   (best 2D)",            0.80, 1.00, 0.50, 0.833, 0.857),
    ("2D-only SVM  (best 2D)",            0.80, 1.00, 0.50, 0.833, 0.857),
    ("2D-only KNN  (worst 2D)",           0.40, 0.67, 0.00, 0.667, 0.571),
]
for name, acc, sens, spec, auc, f1 in refs:
    print(f"  {name:<35} {acc:>4.0%} {sens:>5.0%} {spec:>5.0%} {auc:>6.3f} {f1:>6.3f}")

print("  " + "-"*65)
best_2d3d = max(all_results, key=lambda r: (r[1], r[5] if r[5]==r[5] else 0))
print(f"  {'2D+3D best: '+best_2d3d[0]:<35} {best_2d3d[1]:>4.0%} {best_2d3d[2]:>5.0%} "
      f"{best_2d3d[3]:>5.0%} "
      f"{best_2d3d[4]:>6.3f} {best_2d3d[5]:>6.3f}")

print(f"\n{'='*65}")
print("Done.")
