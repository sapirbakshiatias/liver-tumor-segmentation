"""
Patient-level LOOCV where each SERIES is treated as an independent patient in training.

Protocol:
  1. Feature selection (VaRFS + ANOVA) on ALL data — before any split (same as always)
  2. LOOCV split: held-out patient forms the test set
  3. TRAIN set: every series from the other 4 patients is an INDEPENDENT sample
     - Patient_1 has 11 series  -> 11 independent training examples (all label=1)
     - Patient_2 has  8 series  ->  8 independent training examples (all label=1)
     - Patient_KB has 12 series -> 12 independent training examples (all label=1)
     - Patient_GA has  3 series ->  3 independent training examples (all label=0)
     - (when VT is test, train has ~34 samples instead of 4)
  4. SMOTE to 1:1 balance on the series-level training set
  5. TEST: aggregate series probabilities with 1/rank weighting -> patient prediction

Key difference from original approach (training/train_all_series_report.py):
  - Original: SMOTE on the patient-group level (only 4 training "patients")
  - train_2:  SMOTE on the series level (up to ~34 training series -> more synthetic samples)
  - The classifier sees no patient-grouping metadata during training

Run: .venv\Scripts\python.exe train_2\train_series_as_patients.py
"""
import sys, warnings
sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pipeline.clean_features  import clean_features, cv_filter
from pipeline.compute_icc     import compute_icc
from pipeline.select_features import select_varfs, select_anova
from pipeline.metrics         import metrics
from pipeline.series_weight   import series_weight
from pipeline.models          import make_clf
from imblearn.over_sampling   import SMOTE, RandomOverSampler

CSV_PATH = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\all_series_radiomics.csv"
PATIENTS = ["Patient_1", "Patient_2", "Patient_KB", "Patient_GA", "Patient_VT"]
CLF_NAMES = ["KNN", "RandomForest", "GradientBoost", "SVM",
             "LogisticRegr", "NaiveBayes", "DecisionTree", "MLP"]

# ── Load data ─────────────────────────────────────────────────────────────────

df           = pd.read_csv(CSV_PATH)
feat_cols    = [c for c in df.columns if c not in ("series","patient","group","label")]
groups       = df["group"].tolist()
series_names = df["series"].tolist()
y_all        = df["label"].values.astype(int)
X_raw        = df[feat_cols].values.astype(float)

# ── Feature selection (on ALL data, before any split) ────────────────────────

X_clean, fnames = clean_features(X_raw, list(feat_cols))
cv_keep         = cv_filter(X_clean, fnames)
icc_vals        = compute_icc(df, fnames)

idx_varfs, f_scores, icc_all, _ = select_varfs(X_clean, y_all, fnames, icc_vals, cv_keep)
idx_anova, _                     = select_anova(X_clean, y_all, fnames, cv_keep)

print(f"Loaded {len(df)} series | {len(PATIENTS)} patients | {len(fnames)} clean features")
print(f"\nVaRFS top-5: {[fnames[i] for i in idx_varfs]}")
print(f"ANOVA top-5: {[fnames[i] for i in idx_anova]}")

# ── Augment: SMOTE 1:1 on series-level training set ──────────────────────────

def augment_series(X, y):
    """
    Full 1:1 SMOTE on series-level data.
    With ~34 training series (vs 4 in original), k_neighbors can be larger.
    """
    n_min = (y == 0).sum()
    n_maj = (y == 1).sum()
    if n_min == 0:
        return X, y
    if n_min == 1:
        return RandomOverSampler(random_state=42).fit_resample(X, y)
    k = min(n_min - 1, 5)
    return SMOTE(k_neighbors=k, sampling_strategy=1.0,
                 random_state=42).fit_resample(X, y)


# ── LOOCV: each series in TRAIN is an independent sample ─────────────────────

def loocv(feat_idx, clf_name, weighted=True):
    fold_true, fold_pred, fold_prob = [], [], []
    per_patient = {}

    for patient in PATIENTS:
        test_mask  = np.array([g == patient for g in groups])
        train_mask = ~test_mask

        # TRAIN: each series is an independent example (no patient grouping)
        Xf_tr = X_clean[train_mask][:, feat_idx]
        y_tr  = y_all[train_mask]

        # TEST: all series of the held-out patient
        Xf_te = X_clean[test_mask][:, feat_idx]
        true_label = int(np.round(y_all[test_mask].mean()))

        sc      = StandardScaler()
        Xf_tr_s = sc.fit_transform(Xf_tr)
        Xf_te_s = sc.transform(Xf_te)

        # Augment on series level (more samples -> better SMOTE)
        Xf_aug, y_aug = augment_series(Xf_tr_s, y_tr)

        print(f"    {patient:<15}  train={len(y_aug)} series "
              f"(cancer={int((y_aug==1).sum())}, healthy={int((y_aug==0).sum())}  "
              f"after SMOTE)", end="  ", flush=True)

        clf = make_clf(clf_name)
        clf.fit(Xf_aug, y_aug)
        probs = clf.predict_proba(Xf_te_s)[:, 1]

        if weighted:
            ts  = [series_names[i] for i, m in enumerate(test_mask) if m]
            w   = np.array([series_weight(s) for s in ts])
            avg = float(np.average(probs, weights=w))
        else:
            avg = float(probs.mean())

        pred = int(avg >= 0.5)
        fold_true.append(true_label)
        fold_pred.append(pred)
        fold_prob.append(avg)
        per_patient[patient] = (true_label, pred, avg)

        status = "OK" if true_label==pred else ("FP" if pred==1 and true_label==0 else "FN")
        print(f"P(Cancer)={avg:.3f}  {status}")

    acc, sens, spec, auc, f1 = metrics(fold_true, fold_pred, fold_prob)
    return acc, sens, spec, auc, f1, per_patient


# ── Run all combinations ──────────────────────────────────────────────────────

SEP = "=" * 130
HDR = (f"  {'Model':<28} {'Acc':>5} {'Sens':>5} {'Spec':>5} {'AUC':>6} {'F1':>6} | "
       f"{'P1':>12} {'P2':>12} {'KB':>12} {'GA':>12} {'VT':>12}")


def cell(pp, p):
    tl, pl, prob = pp[p]
    s = "OK" if tl==pl else ("FP" if pl==1 and tl==0 else "FN")
    return f"{s}({prob:.2f})"


for stage_label, weighted in [
    ("STAGE 1 — Simple average (no weighting)", False),
    ("STAGE 2 — Weighted 1/rank",               True),
]:
    print(f"\n{SEP}")
    print(f"  Series-as-Patients LOOCV | {stage_label}")
    print(f"  Each training series is independent — SMOTE 1:1 at series level")
    print(SEP)
    print(HDR)
    print("  " + "-"*128)

    for feat_label, feat_idx in [("VaRFS", idx_varfs), ("ANOVA", idx_anova)]:
        for clf_name in CLF_NAMES:
            print(f"\n  {feat_label}+{clf_name}:")
            acc, sens, spec, auc, f1, pp = loocv(feat_idx, clf_name, weighted)
            auc_s = f"{auc:.3f}" if auc==auc else "  N/A"
            f1_s  = f"{f1:.3f}"  if f1==f1   else "  N/A"
            wrong = [p for p in PATIENTS if pp[p][0] != pp[p][1]]
            errs  = "  <- " + ", ".join(
                f"{p}({'FP' if pp[p][1]==1 and pp[p][0]==0 else 'FN'})"
                for p in wrong) if wrong else ""
            name  = f"{feat_label}+{clf_name}"
            print(f"  {name:<28} {acc:>4.0%} {sens:>5.0%} {spec:>5.0%} "
                  f"{auc_s:>6} {f1_s:>6} | "
                  f"{cell(pp,'Patient_1'):>12} {cell(pp,'Patient_2'):>12} "
                  f"{cell(pp,'Patient_KB'):>12} {cell(pp,'Patient_GA'):>12} "
                  f"{cell(pp,'Patient_VT'):>12}{errs}")

# ── Comparison ────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("  COMPARISON vs Original Approach")
print(SEP)
print(f"  {'Approach':<55} {'Acc':>5} {'Spec':>5} {'F1':>6}")
print("  " + "-"*75)
refs = [
    ("Original: VaRFS+KNN (patient-level SMOTE, 1/rank)",  "100%", "100%", "1.000"),
    ("Original: all others (patient-level SMOTE, 1/rank)", " 80%", " 50%", "0.857"),
    ("Original: all (simple avg, no weighting)",           " 80%", "  0%", "0.857"),
]
for name, acc, spec, f1 in refs:
    print(f"  {name:<55} {acc:>5} {spec:>5} {f1:>6}")
print("  " + "-"*75)
print("  Series-as-Patients results: see table above")
print(SEP)
