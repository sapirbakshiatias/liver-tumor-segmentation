"""
Patient-level LOOCV on U-Net deep features.

Runs the EXACT same experiment pipeline as the 3D radiomics approach:
  - VaRFS + ANOVA feature selection
  - 8 classifiers
  - 1/rank series weighting
  - Comparison against radiomics baseline

The features here come from the U-Net encoder bottleneck (64 values per series),
learned by reconstructing liver CT volumes without any cancer labels.

Run: .venv\Scripts\python.exe training\train_unet_features.py
"""
import sys, warnings
sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from train_all_series_report import (
    clean_features, cv_filter, compute_icc,
    select_varfs, select_anova,
    augment, make_clf, metrics, _series_weight,
)

CSV_PATH = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\unet_features.csv"
PATIENTS = ["Patient_1", "Patient_2", "Patient_KB", "Patient_GA", "Patient_VT"]
CLF_NAMES = ["KNN", "RandomForest", "GradientBoost", "SVM",
             "LogisticRegr", "NaiveBayes", "DecisionTree", "MLP"]

# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)
feat_cols    = [c for c in df.columns if c not in ("series", "patient", "group", "label")]
groups       = df["group"].tolist()
series_names = df["series"].tolist()
y_all        = df["label"].values.astype(int)
X_raw        = df[feat_cols].values.astype(float)

X_clean, fnames = clean_features(X_raw, list(feat_cols))
cv_keep         = cv_filter(X_clean, fnames)
icc_vals        = compute_icc(df, fnames)

idx_varfs, f_scores, icc_all, varfs_scores = select_varfs(
    X_clean, y_all, fnames, icc_vals, cv_keep)
idx_anova, _ = select_anova(X_clean, y_all, fnames, cv_keep)

print(f"U-Net features: {len(df)} series x {len(fnames)} features")
print(f"\nVaRFS top-5 U-Net features:")
for i, idx in enumerate(idx_varfs, 1):
    print(f"  {i}. {fnames[idx]}  F={f_scores[idx]:.2f}  ICC={icc_all[idx]:.3f}")
print(f"\nANOVA top-5 U-Net features:")
for i, idx in enumerate(idx_anova, 1):
    print(f"  {i}. {fnames[idx]}  F={f_scores[idx]:.2f}")

# ── LOOCV ─────────────────────────────────────────────────────────────────────

def loocv(feat_idx, clf_name, weighted):
    fold_true, fold_pred, fold_prob, pp = [], [], [], {}
    for patient in PATIENTS:
        test_mask  = np.array([g == patient for g in groups])
        train_mask = ~test_mask
        Xf_tr = X_clean[train_mask][:, feat_idx]
        Xf_te = X_clean[test_mask][:, feat_idx]
        y_tr  = y_all[train_mask]
        true_label = int(np.round(y_all[test_mask].mean()))

        sc = StandardScaler()
        Xf_tr_s = sc.fit_transform(Xf_tr)
        Xf_te_s = sc.transform(Xf_te)
        Xf_aug, y_aug = augment(Xf_tr_s, y_tr)

        clf = make_clf(clf_name)
        clf.fit(Xf_aug, y_aug)
        probs = clf.predict_proba(Xf_te_s)[:, 1]

        if weighted:
            ts = [series_names[i] for i, m in enumerate(test_mask) if m]
            w  = np.array([_series_weight(s) for s in ts])
            avg_prob = float(np.average(probs, weights=w))
        else:
            avg_prob = float(probs.mean())

        pred = int(avg_prob >= 0.5)
        fold_true.append(true_label); fold_pred.append(pred); fold_prob.append(avg_prob)
        pp[patient] = (true_label, pred, avg_prob)

    acc, sens, spec, auc, f1 = metrics(fold_true, fold_pred, fold_prob)
    return acc, sens, spec, auc, f1, pp


def cell(pp, p):
    tl, pl, prob = pp[p]
    s = "OK" if tl==pl else ("FP" if pl==1 and tl==0 else "FN")
    return f"{s}({prob:.2f})"


# ── Print results ─────────────────────────────────────────────────────────────

SEP = "=" * 130
HDR = (f"  {'Model':<28} {'Acc':>5} {'Sens':>5} {'Spec':>5} {'AUC':>6} {'F1':>6} | "
       f"{'P1':>12} {'P2':>12} {'KB':>12} {'GA':>12} {'VT':>12}")

for stage_label, weighted in [
    ("STAGE 1: Simple average (no weighting)", False),
    ("STAGE 2: Weighted 1/rank",               True),
]:
    print(f"\n{SEP}")
    print(f"  U-Net Features | {stage_label}")
    print(SEP)
    print(HDR)
    print("  " + "-"*128)

    for feat_label, feat_idx in [("VaRFS", idx_varfs), ("ANOVA", idx_anova)]:
        for clf_name in CLF_NAMES:
            acc, sens, spec, auc, f1, pp = loocv(feat_idx, clf_name, weighted)
            auc_s = f"{auc:.3f}" if auc==auc else "  N/A"
            f1_s  = f"{f1:.3f}"  if f1==f1   else "  N/A"
            wrong = [p for p in PATIENTS if pp[p][0] != pp[p][1]]
            errs  = "  <- " + ", ".join(
                f"{p}({'FP' if pp[p][1]==1 and pp[p][0]==0 else 'FN'})"
                for p in wrong) if wrong else ""

            name = f"{feat_label}+{clf_name}"
            print(f"  {name:<28} {acc:>4.0%} {sens:>5.0%} {spec:>5.0%} "
                  f"{auc_s:>6} {f1_s:>6} | "
                  f"{cell(pp,'Patient_1'):>12} {cell(pp,'Patient_2'):>12} "
                  f"{cell(pp,'Patient_KB'):>12} {cell(pp,'Patient_GA'):>12} "
                  f"{cell(pp,'Patient_VT'):>12}{errs}")

# ── Comparison ────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("  COMPARISON: U-Net features vs Radiomics features (best per approach)")
print(SEP)
print(f"  {'Approach':<45} {'Features':<12} {'Acc':>5} {'Spec':>5} {'AUC':>6} {'F1':>6}")
print("  " + "-"*90)

refs = [
    ("3D Radiomics VaRFS+KNN (1/rank)",         "handcrafted", "100%", "100%", "1.000", "1.000"),
    ("2D+3D ANOVA+MLP (thresh=0.45)",            "handcrafted", "100%", "100%", "1.000", "1.000"),
    ("2D VaRFS-5 + MLP (thresh=0.30)",           "handcrafted", "100%", "100%", "1.000", "1.000"),
    ("2D Radiomics RF/SVM",                      "handcrafted", " 80%", " 50%", "0.833", "0.857"),
    ("2D CNN ResNet18",                          "deep (raw)",  " 60%", "  0%", "0.000", "0.750"),
]
for name, ftype, acc, spec, auc, f1 in refs:
    print(f"  {name:<45} {ftype:<12} {acc:>5} {spec:>5} {auc:>6} {f1:>6}")

print("  " + "-"*90)
print(f"  {'U-Net features (best above)':<45} {'deep (AE)':<12} see table above")
print(SEP)
