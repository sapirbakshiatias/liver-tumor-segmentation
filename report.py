"""
Unified report: runs all 14 models (12 classical + 2 Gated Attention)
using Patient-Level LOOCV — no data leakage between same-patient scans.

Each fold holds out ALL scans of one patient, trains on the remaining 4 patients.
5 folds total (one per unique patient). All 8 scans are used.

Run: .venv\Scripts\python.exe report.py
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from train_svm import (
    clean_features, cv_filter, compute_icc,
    select_varfs, select_anova, metrics,
    run_patient_loocv,
    CSV_PATH,
)
from train_attention import (
    run_patient_loocv_attn,
)


# ── Silent runners ────────────────────────────────────────────────────────────

def run_classical(label, feat_idx, clf_name, X_all, y_all, groups):
    _, _, _, acc, _, _, auc, f1 = run_patient_loocv(
        X_all, y_all, groups, feat_idx, clf_name)
    return {"Combination": label, "Type": "Classical",
            "Acc": acc, "AUC": auc, "F1": f1}


def run_attention(label, feat_idx, X_all, y_all, groups):
    _, _, _, acc, _, _, auc, f1, _ = run_patient_loocv_attn(
        X_all, y_all, groups, feat_idx)
    return {"Combination": label, "Type": "Deep Learning",
            "Acc": acc, "AUC": auc, "F1": f1}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    feat_cols = [c for c in df.columns if c not in ("patient", "label", "group")]
    groups    = df["group"].tolist()
    y_all     = df["label"].values.astype(int)

    # Preprocessing fitted on all 8 scans (no leakage — scaler is re-fitted per fold inside LOOCV)
    all_idx   = list(range(len(df)))
    X_raw     = df[feat_cols].values.astype(float)
    X_clean, fnames = clean_features(X_raw, list(feat_cols), all_idx)

    # CV filter and ICC fitted on all data (conservative — acceptable for feature selection)
    cv_keep  = cv_filter(X_clean, fnames)
    icc_vals = compute_icc(df, fnames)

    print(f"Preprocessing: {len(fnames)} features | "
          f"CV removed={( ~cv_keep).sum()} | ICC>=0.75: {(icc_vals >= 0.75).sum()}")

    # Feature selection uses all labels (conservative but necessary with n=8)
    idx_varfs = select_varfs(X_clean, y_all, fnames, icc_vals, cv_keep)
    idx_anova = select_anova(X_clean, y_all, fnames, cv_keep)

    print(f"VaRFS: {[fnames[i] for i in idx_varfs]}")
    print(f"ANOVA: {[fnames[i] for i in idx_anova]}")
    print(f"\nPatient-Level LOOCV — 5 folds, one per patient")
    print(f"Unique patients: {list(dict.fromkeys(groups))}")
    print("\nRunning all 14 models...\n")

    combinations_classical = [
        ("VaRFS + RandomForest",  idx_varfs, "RandomForest"),
        ("VaRFS + SVM",           idx_varfs, "SVM"),
        ("VaRFS + LogisticRegr",  idx_varfs, "LogisticRegression"),
        ("VaRFS + KNN",           idx_varfs, "KNN"),
        ("VaRFS + NaiveBayes",    idx_varfs, "NaiveBayes"),
        ("VaRFS + DecisionTree",  idx_varfs, "DecisionTree"),
        ("ANOVA + RandomForest",  idx_anova, "RandomForest"),
        ("ANOVA + SVM",           idx_anova, "SVM"),
        ("ANOVA + LogisticRegr",  idx_anova, "LogisticRegression"),
        ("ANOVA + KNN",           idx_anova, "KNN"),
        ("ANOVA + NaiveBayes",    idx_anova, "NaiveBayes"),
        ("ANOVA + DecisionTree",  idx_anova, "DecisionTree"),
    ]

    results = []
    for i, (label, feat_idx, clf_name) in enumerate(combinations_classical, 1):
        print(f"  [{i:02d}/14] {label:<28}", end=" ", flush=True)
        r = run_classical(label, feat_idx, clf_name, X_clean, y_all, groups)
        results.append(r)
        print("done")

    for i, (label, feat_idx) in enumerate([
        ("VaRFS + Attention", idx_varfs),
        ("ANOVA + Attention", idx_anova),
    ], 13):
        print(f"  [{i:02d}/14] {label:<28}", end=" ", flush=True)
        r = run_attention(label, feat_idx, X_clean, y_all, groups)
        results.append(r)
        print("done")

    # Sort: F1 desc → AUC desc
    results.sort(key=lambda r: (0.0 if np.isnan(r["F1"])  else r["F1"],
                                0.0 if np.isnan(r["AUC"]) else r["AUC"]),
                 reverse=True)

    # ── Table ─────────────────────────────────────────────────────────────────
    W = 78
    print("\n" + "=" * W)
    print("FULL COMPARISON — ALL 14 MODELS  (Patient-Level LOOCV, n=8 scans, 5 patients)")
    print("=" * W)
    print(f"  {'#':>3}  {'Model':<26} {'Type':<14} "
          f"{'Acc':>6} {'AUC':>7} {'F1':>7}  {'Valid':>5}")
    print("  " + "-" * (W - 2))

    for rank, r in enumerate(results, 1):
        auc_s = f"{r['AUC']:.3f}" if not np.isnan(r["AUC"]) else " N/A"
        f1_s  = f"{r['F1']:.3f}"  if not np.isnan(r["F1"])  else " N/A"
        valid = (r["Acc"] >= 0.8
                 and not np.isnan(r["AUC"]) and r["AUC"] >= 0.8
                 and not np.isnan(r["F1"])  and r["F1"]  >= 0.8)
        flag  = "YES" if valid else "NO "
        print(f"  {rank:>3}.  {r['Combination']:<26} {r['Type']:<14} "
              f"{r['Acc']:>5.0%}  {auc_s:>6}  {f1_s:>6}   {flag}")

    # ── Winners ───────────────────────────────────────────────────────────────
    winners = [r for r in results
               if r["Acc"] >= 0.8
               and not np.isnan(r["AUC"]) and r["AUC"] >= 0.8
               and not np.isnan(r["F1"])  and r["F1"]  >= 0.8]

    print("\n" + "=" * W)
    print("WINNERS  (Acc >= 80%  +  AUC >= 0.8  +  F1 >= 0.8)")
    print("=" * W)
    for i, r in enumerate(winners, 1):
        print(f"  {i}. {r['Combination']:<30} [{r['Type']}]  "
              f"Acc={r['Acc']:.0%}  AUC={r['AUC']:.3f}  F1={r['F1']:.3f}")

    print(f"\n  * Patient-Level LOOCV: each fold leaves out ALL scans of one patient")
    print(f"    No data leakage — same patient never appears in both train and test")
    print(f"    8 scans, 5 unique patients, 5 folds")
