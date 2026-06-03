"""
Run 2D slice LOOCV using ONLY the top-5 VaRFS features from 3D.

The 5 VaRFS features that achieved 100% in 3D (VaRFS+KNN):
  1. vol_fo_kurtosis               (ICC=0.756, F=8.68)
  2. vol_fo_p10                    (ICC=0.179, F=18.25)
  3. vol_fo_mean                   (ICC=0.184, F=17.37)
  4. vol_sagittal_glcm_dissimilarity (ICC=0.573, F=5.50)
  5. vol_sagittal_glcm_contrast    (ICC=0.416, F=5.09)

These are the 3D volume features projected onto each slice from slice_2d_plus_3d.csv.
Each feature is CONSTANT across all slices of a series.

Question: does the number of data points (795 slices vs 28 series)
help or hurt when the features themselves carry no slice-level variation?

Run: .venv\Scripts\python.exe training\train_2d_varfs5.py
"""
import sys, warnings
sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler

from pipeline.metrics import metrics

CSV_PATH = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\slice_2d_plus_3d.csv"

# The exact 5 VaRFS features from the 3D analysis (as vol_ columns in the combined CSV)

# In slice_2d_plus_3d.csv:
# - fo_* features overlap with 2D → renamed to vol_fo_* (3D version)
# - sagittal_glcm_* did NOT exist in 2D → kept original name (from 3D join)
VARFS_5 = [
    "vol_fo_kurtosis",
    "vol_fo_p10",
    "vol_fo_mean",
    "sagittal_glcm_dissimilarity",   # no vol_ prefix — unique to 3D
    "sagittal_glcm_contrast",        # no vol_ prefix — unique to 3D
]

# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=VARFS_5)
df = df[np.all(np.isfinite(df[VARFS_5].values), axis=1)]

groups   = df["group"].tolist()
y_all    = df["label"].values.astype(int)
X_all    = df[VARFS_5].values.astype(float)
PATIENTS = list(dict.fromkeys(groups))

n_cancer  = (y_all == 1).sum()
n_healthy = (y_all == 0).sum()

print("=" * 70)
print("2D SLICES — Top-5 VaRFS features from 3D (vol_ columns)")
print("=" * 70)
print(f"Slices: {len(df)}  |  Cancer: {n_cancer}  |  Healthy: {n_healthy}")
print(f"\nFeatures used (exactly the VaRFS top-5 from the 3D pipeline):")
for i, f in enumerate(VARFS_5, 1):
    clean = f.replace("vol_", "")
    print(f"  {i}. {f}  (3D: {clean})")

# ── Helpers ───────────────────────────────────────────────────────────────────

def augment(X, y):
    n = (y == 0).sum()
    if n >= 2:
        return SMOTE(k_neighbors=min(n-1, 5), sampling_strategy=1.0,
                     random_state=42).fit_resample(X, y)
    if n == 1:
        return RandomOverSampler(random_state=42).fit_resample(X, y)
    return X, y


def make_clf(name):
    return {
        "KNN":           KNeighborsClassifier(n_neighbors=3),
        "RandomForest":  RandomForestClassifier(200, class_weight="balanced", random_state=42),
        "GradientBoost": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
        "SVM":           SVC(kernel="rbf", C=1.0, class_weight="balanced",
                             probability=True, random_state=42),
        "MLP":           MLPClassifier((64, 32), max_iter=500, random_state=42,
                                       early_stopping=True, validation_fraction=0.1),
    }[name]


def loocv(clf_name, threshold=0.5):
    pat_probs = {}
    for patient in PATIENTS:
        test_mask  = np.array([g == patient for g in groups])
        train_mask = ~test_mask

        X_tr = X_all[train_mask]
        X_te = X_all[test_mask]
        y_tr = y_all[train_mask]
        true_label = int(y_all[test_mask][0])

        sc      = StandardScaler()
        X_tr_s  = sc.fit_transform(X_tr)
        X_te_s  = sc.transform(X_te)
        X_aug, y_aug = augment(X_tr_s, y_tr)

        clf = make_clf(clf_name)
        clf.fit(X_aug, y_aug)
        slice_probs = clf.predict_proba(X_te_s)[:, 1]

        # Weight: liver_px × (1/series_rank) — same as the 3D fix
        lx      = df[test_mask]["liver_px"].values.astype(float)
        series  = df[test_mask]["series"].values
        rank_w  = np.array([1.0 / int(s.split("_s")[-1]) for s in series])
        weights = (lx / lx.max()) * rank_w
        avg_prob = float(np.average(slice_probs, weights=weights))

        pat_probs[patient] = (true_label, avg_prob)

    y_true = [pat_probs[p][0] for p in PATIENTS]
    y_prob = [pat_probs[p][1] for p in PATIENTS]
    y_pred = [int(pr >= threshold) for pr in y_prob]
    pp     = {p: (pat_probs[p][0], int(pat_probs[p][1] >= threshold), pat_probs[p][1])
              for p in PATIENTS}
    acc, sens, spec, auc, f1 = metrics(y_true, y_pred, y_prob)
    return acc, sens, spec, auc, f1, pp


# ── Threshold sweep ───────────────────────────────────────────────────────────

THRESHOLDS = np.arange(0.20, 0.55, 0.05)
CLF_NAMES  = ["KNN", "RandomForest", "GradientBoost", "SVM", "MLP"]

def cell(pp, p):
    tl, pl, prob = pp[p]
    s = "OK" if tl==pl else ("FP" if pl==1 and tl==0 else "FN")
    return f"{s}({prob:.2f})"

print(f"\n{'Model':<16} {'Thr':>5} {'Acc':>5} {'Sens':>5} {'Spec':>5} {'AUC':>6} {'F1':>6} | "
      f"{'P1':>10} {'P2':>10} {'KB':>10} {'GA':>10} {'VT':>10}")
print("-" * 115)

all_results = []

for clf_name in CLF_NAMES:
    best_acc, best_row = -1, None

    for thresh in THRESHOLDS:
        acc, sens, spec, auc, f1, pp = loocv(clf_name, threshold=thresh)
        score = (acc, spec if spec==spec else 0)
        if score > (best_acc, 0):
            best_acc = acc
            best_row = (thresh, acc, sens, spec, auc, f1, pp)

    thresh, acc, sens, spec, auc, f1, pp = best_row
    auc_s = f"{auc:.3f}" if auc==auc else "  N/A"
    f1_s  = f"{f1:.3f}"  if f1==f1   else "  N/A"
    wrong = [p for p in PATIENTS if pp[p][0] != pp[p][1]]
    errs  = "  <- " + ", ".join(
        f"{p}({'FP' if pp[p][1]==1 and pp[p][0]==0 else 'FN'})" for p in wrong
    ) if wrong else ""

    print(f"{clf_name:<16} {thresh:>5.2f} {acc:>4.0%} {sens:>5.0%} {spec:>5.0%} "
          f"{auc_s:>6} {f1_s:>6} | "
          f"{cell(pp,'Patient_1'):>10} {cell(pp,'Patient_2'):>10} "
          f"{cell(pp,'Patient_KB'):>10} {cell(pp,'Patient_GA'):>10} "
          f"{cell(pp,'Patient_VT'):>10}{errs}")
    all_results.append((clf_name, thresh, acc, sens, spec, auc, f1, pp))

# ── Threshold sweep detail for best model ────────────────────────────────────

best = max(all_results, key=lambda r: (r[2], r[4] if r[4]==r[4] else 0))
print(f"\nBest: {best[0]}  threshold={best[1]:.2f}  Acc={best[2]:.0%}")

print(f"\nThreshold sweep for {best[0]}:")
print(f"  {'Thresh':>7} {'Acc':>5} {'Sens':>5} {'Spec':>5} {'F1':>6}  Errors")
print("  " + "-" * 60)
for thresh in THRESHOLDS:
    acc, sens, spec, auc, f1, pp = loocv(best[0], threshold=thresh)
    wrong = [p for p in PATIENTS if pp[p][0] != pp[p][1]]
    errs  = ", ".join(
        f"{p}({'FP' if pp[p][1]==1 and pp[p][0]==0 else 'FN'})" for p in wrong
    ) if wrong else "none"
    f1_s = f"{f1:.3f}" if f1==f1 else " N/A"
    print(f"  {thresh:>7.2f} {acc:>4.0%} {sens:>5.0%} {spec:>5.0%} {f1_s:>6}  {errs}")

# ── Comparison ────────────────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("COMPARISON")
print(f"{'=' * 70}")
print(f"  {'Approach':<40} {'Acc':>5} {'Spec':>5} {'AUC':>6} {'F1':>6}")
print("  " + "-" * 60)
refs = [
    ("3D VaRFS+KNN (top-5, 28 series)",    "100%", "100%", "1.000", "1.000"),
    ("2D+3D ANOVA+MLP (57 features)",      "100%", "100%", "1.000", "1.000"),
    ("2D-only RF (16 features)",           " 80%", " 50%", "0.833", "0.857"),
]
for name, acc, spec, auc, f1 in refs:
    print(f"  {name:<40} {acc:>5} {spec:>5} {auc:>6} {f1:>6}")
print("  " + "-" * 60)
b = max(all_results, key=lambda r: (r[2], r[4] if r[4]==r[4] else 0))
print(f"  {'2D VaRFS-5 best: '+b[0]:<40} {b[2]:>4.0%} "
      f"{b[4]:>5.0%} {b[5]:>6.3f} {b[6]:>6.3f}")

print(f"\nDone.")
