"""
Compare two series-weighting strategies in patient-level LOOCV:
  1. rank-based: weight = 1/rank  (current approach)
  2. volume-based: weight = liver_voxels / max_liver_voxels_of_patient
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")
from train_all_series_report import (
    clean_features, cv_filter, compute_icc, select_varfs, select_anova,
    augment, make_clf, metrics, _series_weight,
)
from sklearn.preprocessing import StandardScaler
import re

CSV_PATH = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\all_series_radiomics.csv"

df           = pd.read_csv(CSV_PATH)
feat_cols    = [c for c in df.columns if c not in ("series","patient","group","label")]
groups       = df["group"].tolist()
series_names = df["series"].tolist()
y_all        = df["label"].values.astype(int)
X_raw        = df[feat_cols].values.astype(float)

X_clean, fnames = clean_features(X_raw, list(feat_cols))
cv_keep         = cv_filter(X_clean, fnames)
icc_vals        = compute_icc(df, fnames)
idx_varfs, _, _, _ = select_varfs(X_clean, y_all, fnames, icc_vals, cv_keep)
idx_anova, _       = select_anova(X_clean, y_all, fnames, cv_keep)

# ── Volume-based weights ──────────────────────────────────────────────────────
# shape_volume is one of the extracted features
vol_col = "shape_volume"
if vol_col in df.columns:
    # normalize per patient: weight = voxels / max_voxels_of_that_patient
    df["vol_weight"] = df.groupby("group")[vol_col].transform(lambda x: x / x.max())
else:
    # fallback: use raw voxel count
    vol_idx = fnames.index("shape_volume") if "shape_volume" in fnames else None
    if vol_idx is not None:
        df["vol_weight"] = X_clean[:, vol_idx]
        df["vol_weight"] = df.groupby("group")["vol_weight"].transform(lambda x: x / x.max())
    else:
        print("shape_volume not found, using uniform weights")
        df["vol_weight"] = 1.0

vol_weights_map = dict(zip(df["series"], df["vol_weight"]))

PATIENTS = ["Patient_1","Patient_2","Patient_KB","Patient_GA","Patient_VT"]
CLF_NAMES = ["RandomForest","SVM","LogisticRegr","KNN","NaiveBayes","DecisionTree"]


def loocv(feat_idx, clf_name, weight_mode):
    fold_true, fold_pred, fold_prob, per_patient = [], [], [], {}
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

        test_series = [series_names[i] for i, m in enumerate(test_mask) if m]
        if weight_mode == "rank":
            weights = np.array([_series_weight(s) for s in test_series])
        elif weight_mode == "volume":
            weights = np.array([vol_weights_map[s] for s in test_series])
        else:
            weights = np.ones(len(test_series))

        avg_prob = float(np.average(probs, weights=weights))
        pred     = int(avg_prob >= 0.5)
        fold_true.append(true_label); fold_pred.append(pred); fold_prob.append(avg_prob)
        per_patient[patient] = (true_label, pred, avg_prob)

    acc, sens, spec, auc, f1 = metrics(fold_true, fold_pred, fold_prob)
    return acc, f1, per_patient


# ── Print volume weights so we can see if they match rank order ───────────────
print("Series volume weights (per patient, normalized):")
for grp, sub in df.groupby("group"):
    print(f"\n  {grp}:")
    for _, row in sub.sort_values("vol_weight", ascending=False).iterrows():
        rank_w = _series_weight(row["series"])
        marker = " <-- rank vs volume DISAGREE" if abs(rank_w - row["vol_weight"]) > 0.25 else ""
        print(f"    {row['series']:<40}  rank_w={rank_w:.3f}  vol_w={row['vol_weight']:.3f}{marker}")

# ── Full comparison ───────────────────────────────────────────────────────────
print("\n\n" + "="*110)
print(f"{'Model':<28} {'Feat':>6} | {'rank Acc':>8} {'rank F1':>8} {'VT(rank)':>10} | "
      f"{'vol Acc':>8} {'vol F1':>8} {'VT(vol)':>10} | Change")
print("-"*110)

for feat_label, feat_idx in [("VaRFS", idx_varfs), ("ANOVA", idx_anova)]:
    for clf_name in CLF_NAMES:
        acc_r, f1_r, pp_r = loocv(feat_idx, clf_name, "rank")
        acc_v, f1_v, pp_v = loocv(feat_idx, clf_name, "volume")

        vt_r = pp_r["Patient_VT"]
        vt_v = pp_v["Patient_VT"]
        vt_r_s = f"{'OK' if vt_r[1]==0 else 'XX'}({vt_r[2]:.2f})"
        vt_v_s = f"{'OK' if vt_v[1]==0 else 'XX'}({vt_v[2]:.2f})"

        change = ""
        if acc_r != acc_v:
            change = f"Acc {acc_r:.0%}->{acc_v:.0%}"
        if vt_r[1] != vt_v[1]:
            change += f"  VT flips!"

        print(f"{feat_label}+{clf_name:<22} | {acc_r:>7.0%} {f1_r:>8.3f} {vt_r_s:>10} | "
              f"{acc_v:>7.0%} {f1_v:>8.3f} {vt_v_s:>10} | {change}")
