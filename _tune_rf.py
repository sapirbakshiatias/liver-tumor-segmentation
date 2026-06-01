"""
Quick RF hyperparameter sweep to see if we can fix VT without breaking others.
Tests combinations of min_samples_leaf and max_depth in patient-level LOOCV.
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")
from train_all_series_report import (
    clean_features, cv_filter, compute_icc, select_varfs, augment,
    _series_weight, metrics
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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

PATIENTS = ["Patient_1", "Patient_2", "Patient_KB", "Patient_GA", "Patient_VT"]

def loocv_rf(n_est, max_depth, min_leaf, max_feat):
    fold_true, fold_pred, fold_prob, per_patient = [], [], [], {}
    for patient in PATIENTS:
        test_mask  = np.array([g == patient for g in groups])
        train_mask = ~test_mask
        Xf_tr = X_clean[train_mask][:, idx_varfs]
        Xf_te = X_clean[test_mask][:, idx_varfs]
        y_tr  = y_all[train_mask]
        true_label = int(np.round(y_all[test_mask].mean()))

        sc = StandardScaler()
        Xf_tr_s = sc.fit_transform(Xf_tr)
        Xf_te_s = sc.transform(Xf_te)
        Xf_aug, y_aug = augment(Xf_tr_s, y_tr)

        clf = RandomForestClassifier(
            n_estimators=n_est, max_depth=max_depth,
            min_samples_leaf=min_leaf, max_features=max_feat,
            class_weight="balanced", random_state=42
        )
        clf.fit(Xf_aug, y_aug)
        probs = clf.predict_proba(Xf_te_s)[:, 1]

        test_series = [series_names[i] for i, m in enumerate(test_mask) if m]
        weights     = np.array([_series_weight(s) for s in test_series])
        avg_prob    = float(np.average(probs, weights=weights))
        pred        = int(avg_prob >= 0.5)

        fold_true.append(true_label); fold_pred.append(pred); fold_prob.append(avg_prob)
        per_patient[patient] = (true_label, pred, avg_prob)

    acc, sens, spec, auc, f1 = metrics(fold_true, fold_pred, fold_prob)
    return acc, sens, spec, auc, f1, per_patient


print(f"{'n_est':>6} {'depth':>6} {'leaf':>5} {'feat':>5} | {'Acc':>5} {'Sens':>5} {'Spec':>5} {'AUC':>6} | "
      f"P1    P2    KB    GA    VT")
print("-" * 100)

configs = [
    # baseline
    (100, None, 1,  "sqrt"),
    # more trees, same rest
    (300, None, 1,  "sqrt"),
    # limit depth
    (200, 4,    1,  "sqrt"),
    (200, 5,    1,  "sqrt"),
    (200, 6,    1,  "sqrt"),
    # smooth leaves
    (200, None, 2,  "sqrt"),
    (200, None, 3,  "sqrt"),
    (200, None, 4,  "sqrt"),
    (200, None, 5,  "sqrt"),
    # combine depth + leaf
    (200, 5,    2,  "sqrt"),
    (200, 5,    3,  "sqrt"),
    (200, 4,    2,  "sqrt"),
    (200, 4,    3,  "sqrt"),
    # more features
    (200, None, 2,  None),   # max_features=None -> all 5
    (200, 5,    2,  None),
]

for (n_est, depth, leaf, feat) in configs:
    acc, sens, spec, auc, f1, pp = loocv_rf(n_est, depth, leaf, feat)
    depth_s = str(depth) if depth else "None"
    feat_s  = str(feat)  if feat  else "all"
    auc_s   = f"{auc:.3f}" if auc == auc else "N/A"

    row_pp = ""
    for p in PATIENTS:
        tl, pl, pr = pp[p]
        ok = "OK" if tl == pl else "XX"
        row_pp += f"  {ok}({pr:.2f})"

    marker = " <-- VT fixed!" if pp["Patient_VT"][1] == 0 else ""
    print(f"{n_est:>6} {depth_s:>6} {leaf:>5} {feat_s:>5} | "
          f"{acc:>4.0%} {sens:>4.0%} {spec:>4.0%} {auc_s:>6} |{row_pp}{marker}")
