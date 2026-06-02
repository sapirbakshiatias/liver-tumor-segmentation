"""
Patient-Level LOOCV — Main runner.

מייבא את כל הלוגיקה מתיקיית pipeline/ ומריץ את ה-pipeline המלא.

Run: .venv\Scripts\python.exe train_all_series_report.py
"""

import sys, os; sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")


# ── Re-exports (שאר הקבצים מייבאים מכאן) ────────────────────────────────────
from pipeline.data_preparation import (
    clean_features, cv_filter, compute_icc,
    select_varfs, select_anova,
    augment, metrics,
    _series_weight, run_patient_loocv,
    load_data, OUT_DIR, N_FEATURES,
)
from pipeline.models import make_clf
from pipeline.visualizations import (
    plot_feature_importance_bar,
    plot_statistical_heatmap,
    plot_interactive_heatmap,
    plot_spatial_heatmaps,
)

import os
import numpy as np

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

    series_names = df["series"].tolist()

    results = []
    for label, feat_idx, clf_name in combinations:
        print(f"  {label:<32}", end=" ", flush=True)
        acc, sens, spec, auc, f1, fold_log = run_patient_loocv(
            X_clean, y_all, groups, feat_idx, clf_name,
            series_names=series_names, weighted=True)
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
    print("SUMMARY TABLE  (sorted by F1 desc)")
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
