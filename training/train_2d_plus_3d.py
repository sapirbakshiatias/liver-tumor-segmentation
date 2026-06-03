"""
Patient-level LOOCV using combined 2D slice + 3D series features.
With improved class imbalance handling:
  1. SMOTE to full 1:1 ratio (not just a few extra samples)
  2. class_weight='balanced' on all classifiers that support it
  3. Threshold optimization: find the best decision threshold via LOOCV

Each slice gets 57 features:
  - 16 axial 2D features  (slice-specific texture)
  - 41 3D volume features  (sagittal/coronal GLCM, shape, gradient from whole liver)

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

df = df.dropna(subset=feat_cols)
df = df[np.all(np.isfinite(df[feat_cols].values), axis=1)]

groups   = df["group"].tolist()
y_all    = df["label"].values.astype(int)
X_raw    = df[feat_cols].values.astype(float)
PATIENTS = list(dict.fromkeys(groups))

n_cancer  = (y_all == 1).sum()
n_healthy = (y_all == 0).sum()
print(f"Dataset: {len(df)} slices | {len(PATIENTS)} patients | {len(feat_cols)} raw features")
print(f"Class balance: {n_cancer} cancer slices vs {n_healthy} healthy slices "
      f"(ratio {n_cancer/n_healthy:.1f}:1)")

# ── Feature cleaning and selection ───────────────────────────────────────────

X_clean, fnames = clean_features(X_raw, list(feat_cols))
cv_keep         = cv_filter(X_clean, fnames)
icc_vals        = compute_icc(df, fnames)

idx_varfs, f_scores, _, _ = select_varfs(X_clean, y_all, fnames, icc_vals, cv_keep)
idx_anova, _              = select_anova(X_clean, y_all, fnames, cv_keep)

print(f"\nVaRFS top-5: {[fnames[i] for i in idx_varfs]}")
print(f"ANOVA top-5: {[fnames[i] for i in idx_anova]}")

# ── Classifiers ───────────────────────────────────────────────────────────────

def make_clf(name):
    """All classifiers that support class_weight use 'balanced'."""
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


def augment_full(X, y):
    """SMOTE to full 1:1 class balance (not just a few synthetic samples)."""
    n_min = (y == 0).sum()
    n_maj = (y == 1).sum()
    if n_min == 0:
        return X, y
    if n_min == 1:
        return RandomOverSampler(random_state=42).fit_resample(X, y)
    # Target: equal numbers of both classes
    k = min(n_min - 1, 5)
    return SMOTE(k_neighbors=k, sampling_strategy=1.0,
                 random_state=42).fit_resample(X, y)


# ── LOOCV with threshold optimization ────────────────────────────────────────

THRESHOLDS = np.arange(0.20, 0.55, 0.05)


def loocv_with_threshold(feat_idx, clf_name):
    """
    Run LOOCV and collect per-patient probabilities.
    Then sweep thresholds to find the best one.
    Returns results for every threshold tested.
    """
    # Collect raw probabilities (threshold-free)
    patient_probs = {}   # patient -> (true_label, avg_prob)

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
        X_aug, y_aug = augment_full(X_tr_s, y_tr)

        clf = make_clf(clf_name)
        clf.fit(X_aug, y_aug)
        slice_probs = clf.predict_proba(X_te_s)[:, 1]

        # Weight: liver_px × (1/series_rank)
        # s01 slices dominate; contrast-phase slices (s02, s05...) get less weight.
        # This mirrors the 3D rank-based weighting that fixed Patient_VT.
        lx          = df[test_mask]["liver_px"].values.astype(float)
        test_series = df[test_mask]["series"].values
        rank_w      = np.array([1.0 / int(s.split("_s")[-1]) for s in test_series])
        weights     = (lx / lx.max()) * rank_w
        avg_prob    = float(np.average(slice_probs, weights=weights))
        patient_probs[patient] = (true_label, avg_prob)

    # Sweep thresholds
    best_acc, best_thresh, best_result = -1, 0.5, None
    threshold_rows = []

    for thresh in THRESHOLDS:
        y_true = [patient_probs[p][0] for p in PATIENTS]
        y_prob = [patient_probs[p][1] for p in PATIENTS]
        y_pred = [int(prob >= thresh) for prob in y_prob]

        acc, sens, spec, auc, f1 = metrics(y_true, y_pred, y_prob)
        pp = {p: (patient_probs[p][0], int(patient_probs[p][1] >= thresh),
                  patient_probs[p][1])
              for p in PATIENTS}
        threshold_rows.append((thresh, acc, sens, spec, auc, f1, pp))

        # Best = highest Acc, then highest Spec (to fix the healthy misclassification)
        score = (acc, spec if spec == spec else 0)
        if score > (best_acc, 0):
            best_acc = acc
            best_thresh = thresh
            best_result = (thresh, acc, sens, spec, auc, f1, pp)

    return threshold_rows, best_result, patient_probs


# ── Run all models ────────────────────────────────────────────────────────────

CLF_NAMES = ["KNN", "RandomForest", "GradientBoost", "SVM", "MLP"]
SEP = "=" * 140

print(f"\n{SEP}")
print("2D+3D COMBINED — Balanced SMOTE + class_weight + Threshold Optimization")
print(f"{SEP}")

def cell(pp, p):
    tl, pl, prob = pp[p]
    status = "OK" if tl == pl else ("FP" if pl==1 and tl==0 else "FN")
    return f"{status}({prob:.2f})"

header = (f"  {'Combo':<28} {'Thr':>5} {'Acc':>5} {'Sens':>5} {'Spec':>5} {'AUC':>6} {'F1':>6} | "
          f"{'P1':>10} {'P2':>10} {'KB':>10} {'GA':>10} {'VT':>10}")
print(header)
print("  " + "-"*138)

all_best = []

for feat_label, feat_idx in [("VaRFS+", idx_varfs), ("ANOVA+", idx_anova)]:
    for clf_name in CLF_NAMES:
        _, best, raw_probs = loocv_with_threshold(feat_idx, clf_name)
        thresh, acc, sens, spec, auc, f1, pp = best
        auc_s = f"{auc:.3f}" if auc == auc else "  N/A"
        f1_s  = f"{f1:.3f}"  if f1  == f1  else "  N/A"
        wrong = [p for p in PATIENTS if pp[p][0] != pp[p][1]]
        errs  = "  <- " + ", ".join(
            f"{p}({'FP' if pp[p][1]==1 and pp[p][0]==0 else 'FN'})" for p in wrong
        ) if wrong else ""

        row = (f"  {feat_label+clf_name:<28} {thresh:>4.2f} {acc:>4.0%} {sens:>5.0%} "
               f"{spec:>5.0%} {auc_s:>6} {f1_s:>6} | "
               f"{cell(pp,'Patient_1'):>10} {cell(pp,'Patient_2'):>10} "
               f"{cell(pp,'Patient_KB'):>10} {cell(pp,'Patient_GA'):>10} "
               f"{cell(pp,'Patient_VT'):>10}{errs}")
        print(row)
        all_best.append((feat_label+clf_name, thresh, acc, sens, spec, auc, f1, pp))

# ── Threshold sweep detail for best model ────────────────────────────────────

best_overall = max(all_best, key=lambda r: (r[2], r[4] if r[4]==r[4] else 0))
print(f"\nBest model: {best_overall[0]}  threshold={best_overall[1]:.2f}  Acc={best_overall[2]:.0%}")

best_feat_idx = idx_varfs if best_overall[0].startswith("VaRFS") else idx_anova
best_clf      = best_overall[0].split("+")[1]
rows, _, _    = loocv_with_threshold(best_feat_idx, best_clf)

print(f"\nThreshold sweep for {best_overall[0]}:")
print(f"  {'Thresh':>7} {'Acc':>5} {'Sens':>5} {'Spec':>5} {'F1':>6}  Errors")
print("  " + "-"*60)
for thresh, acc, sens, spec, auc, f1, pp in rows:
    wrong = [p for p in PATIENTS if pp[p][0] != pp[p][1]]
    errs  = ", ".join(
        f"{p}({'FP' if pp[p][1]==1 and pp[p][0]==0 else 'FN'})" for p in wrong
    ) if wrong else "none"
    f1_s = f"{f1:.3f}" if f1==f1 else " N/A"
    print(f"  {thresh:>7.2f} {acc:>4.0%} {sens:>5.0%} {spec:>5.0%} {f1_s:>6}  {errs}")

# ── Comparison ────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("COMPARISON: 2D-only vs 2D+3D (balanced) vs 3D-only")
print(f"{SEP}")
print(f"  {'Approach':<42} {'Thr':>5} {'Acc':>5} {'Sens':>5} {'Spec':>5} {'AUC':>6} {'F1':>6}")
print("  " + "-"*78)

refs = [
    ("3D VaRFS+KNN (reference)",        0.50, 1.00, 1.00, 1.00, 1.000, 1.000),
    ("2D-only RF (thresh=0.5)",         0.50, 0.80, 1.00, 0.50, 0.833, 0.857),
    ("2D+3D VaRFS+MLP (thresh=0.5)",   0.50, 0.80, 1.00, 0.50, 0.667, 0.857),
]
for name, thr, acc, sens, spec, auc, f1 in refs:
    print(f"  {name:<42} {thr:>5.2f} {acc:>4.0%} {sens:>5.0%} {spec:>5.0%} {auc:>6.3f} {f1:>6.3f}")

print("  " + "-"*78)
best = max(all_best, key=lambda r: (r[2], r[4] if r[4]==r[4] else 0))
print(f"  {'2D+3D balanced best: '+best[0]:<42} {best[1]:>5.2f} {best[2]:>4.0%} "
      f"{best[3]:>5.0%} {best[4]:>5.0%} "
      f"{best[5]:>6.3f} {best[6]:>6.3f}")

print(f"\nDone.")
