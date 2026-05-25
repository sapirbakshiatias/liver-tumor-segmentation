"""
Full pipeline with comparison: VaRFS vs ANOVA  x  RandomForest vs SVM
4 combinations run in parallel, results compared in a summary table.

Pipeline per combination:
  1. Data cleaning (constant/NaN removal, outlier clipping)
  2. CV filter (remove high-variance noisy features)
  3. Feature selection: VaRFS (F-score x ICC) OR ANOVA (F-score only)
  4. LOOCV on 5 train patients + SMOTE → accuracy, sensitivity, specificity, AUC
  5. Final model trained on all train data → evaluated on 3 test patients
"""

import warnings
import numpy as np
import pandas as pd
import sklearn.base

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

from imblearn.over_sampling import SMOTE, RandomOverSampler

warnings.filterwarnings("ignore")

CSV_PATH = r"c:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\liver_radiomics.csv"

ICC_PAIR_NAMES = [
    ("Patient_1_Before_CT",  "Patient_1_After_CT"),
    ("Patient_2_Before_CT",  "Patient_2_After_CT"),
    ("Patient_KB_Before_CT", "Patient_KB_Cancer_CT"),
]

CV_THRESHOLD  = 100.0
N_FEATURES    = 5
RF_ESTIMATORS = 100


# ── Data cleaning ─────────────────────────────────────────────────────────────

def clean_features(X_all, feature_names, train_idx):
    X = X_all.copy().astype(float)
    bad = np.any(~np.isfinite(X), axis=0)
    X = X[:, ~bad]
    feature_names = [f for f, b in zip(feature_names, bad) if not b]

    train_std = X[train_idx].std(axis=0)
    const = train_std == 0
    X = X[:, ~const]
    feature_names = [f for f, c in zip(feature_names, const) if not c]

    mu = X[train_idx].mean(axis=0)
    sd = X[train_idx].std(axis=0)
    X  = np.clip(X, mu - 3 * sd, mu + 3 * sd)
    return X, feature_names


# ── CV filter ─────────────────────────────────────────────────────────────────

def cv_filter(X_train, feature_names):
    mean = np.abs(X_train.mean(axis=0))
    std  = X_train.std(axis=0)
    cv   = np.where(mean > 1e-10, std / mean * 100, np.inf)
    return cv <= CV_THRESHOLD


# ── ICC ───────────────────────────────────────────────────────────────────────

def compute_icc(df, feature_names):
    before_rows, after_rows = [], []
    for b_name, a_name in ICC_PAIR_NAMES:
        b = df[df["patient"] == b_name]
        a = df[df["patient"] == a_name]
        if b.empty or a.empty:
            continue
        before_rows.append(b[feature_names].values[0])
        after_rows.append(a[feature_names].values[0])

    if len(before_rows) < 2:
        return np.ones(len(feature_names))

    B = np.array(before_rows)
    A = np.array(after_rows)
    n, k = B.shape[0], 2
    vals = np.stack([B, A], axis=1)
    subj_means  = vals.mean(axis=1)
    grand_means = vals.mean(axis=(0, 1))

    MSB = k * np.sum((subj_means - grand_means) ** 2, axis=0) / (n - 1)
    MSW = np.sum((vals - subj_means[:, None, :]) ** 2, axis=(0, 1)) / (n * (k - 1))
    denom = MSB + (k - 1) * MSW
    icc = np.where(denom > 0, (MSB - MSW) / denom, 0.0)
    return np.clip(icc, 0.0, 1.0)


# ── Feature selection ─────────────────────────────────────────────────────────

def select_varfs(X_train, y_train, feature_names, icc_vals, cv_keep, k=N_FEATURES):
    """VaRFS: combined score = normalised F-score × ICC (soft penalty)."""
    f_scores, _ = f_classif(X_train, y_train)
    f_scores = np.nan_to_num(f_scores)
    f_norm   = f_scores / (f_scores.max() + 1e-10)
    score    = f_norm * icc_vals
    score[~cv_keep] = 0.0
    return np.argsort(score)[::-1][:k]


def select_anova(X_train, y_train, feature_names, cv_keep, k=N_FEATURES):
    """ANOVA: select top-k by F-score only (no stability weighting)."""
    f_scores, _ = f_classif(X_train, y_train)
    f_scores = np.nan_to_num(f_scores)
    f_scores[~cv_keep] = 0.0
    return np.argsort(f_scores)[::-1][:k]


# ── Augmentation + metrics ────────────────────────────────────────────────────

def augment(X, y):
    n_min = (y == 0).sum()
    if n_min >= 2:
        return SMOTE(k_neighbors=1, random_state=42).fit_resample(X, y)
    if n_min == 1:
        return RandomOverSampler(random_state=42).fit_resample(X, y)
    return X, y


def metrics(y_true, y_pred, y_prob=None):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc  = (tp + tn) / len(y_true)
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1   = (2 * prec * sens / (prec + sens)
            if not (np.isnan(prec) or np.isnan(sens)) and (prec + sens) > 0
            else float("nan"))
    auc  = float("nan")
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            pass
    return acc, sens, spec, auc, prec, f1


def make_clf(clf_name):
    if clf_name == "RandomForest":
        return RandomForestClassifier(n_estimators=RF_ESTIMATORS,
                                      class_weight="balanced", random_state=42)
    if clf_name == "SVM":
        return SVC(kernel="rbf", C=1.0, class_weight="balanced",
                   probability=True, random_state=42)
    if clf_name == "LogisticRegression":
        return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    if clf_name == "KNN":
        return KNeighborsClassifier(n_neighbors=3)
    if clf_name == "NaiveBayes":
        return GaussianNB()
    if clf_name == "DecisionTree":
        return DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
    raise ValueError(f"Unknown classifier: {clf_name}")


# ── LOOCV ─────────────────────────────────────────────────────────────────────

def run_loocv(X_tr, y_tr, feat_idx, clf_name):
    loo = LeaveOneOut()
    val_true, val_pred, val_prob, tr_accs = [], [], [], []

    for train_idx, test_idx in loo.split(X_tr):
        Xf_tr = X_tr[train_idx][:, feat_idx]
        Xf_te = X_tr[test_idx][:, feat_idx]
        y_fold = y_tr[train_idx]

        scaler   = StandardScaler()
        Xf_tr_s  = scaler.fit_transform(Xf_tr)
        Xf_te_s  = scaler.transform(Xf_te)

        Xf_aug, y_aug = augment(Xf_tr_s, y_fold)

        if len(np.unique(y_aug)) < 2:
            only_class = int(y_aug[0])
            tr_accs.append(1.0)
            val_true.append(int(y_tr[test_idx][0]))
            val_pred.append(only_class)
            val_prob.append(float(only_class))
            continue

        clf = make_clf(clf_name)
        clf.fit(Xf_aug, y_aug)

        tr_accs.append(clf.score(Xf_aug, y_aug))
        val_true.append(int(y_tr[test_idx][0]))
        val_pred.append(int(clf.predict(Xf_te_s)[0]))
        proba = clf.predict_proba(Xf_te_s)
        val_prob.append(float(proba[0, 1]) if proba.shape[1] == 2 else float(clf.classes_[0]))

    acc, sens, spec, auc, prec, f1 = metrics(val_true, val_pred, val_prob)
    return val_true, val_pred, val_prob, float(np.mean(tr_accs)), acc, sens, spec, auc, f1


def run_patient_loocv(X_all, y_all, groups, feat_idx, clf_name):
    """Leave-one-PATIENT-out: all scans of one patient held out together."""
    unique_patients = list(dict.fromkeys(groups))
    val_true, val_pred, val_prob = [], [], []

    for patient in unique_patients:
        test_mask  = np.array([g == patient for g in groups])
        train_mask = ~test_mask

        Xf_tr  = X_all[train_mask][:, feat_idx]
        Xf_te  = X_all[test_mask][:, feat_idx]
        y_fold = y_all[train_mask]
        y_test = y_all[test_mask]

        scaler   = StandardScaler()
        Xf_tr_s  = scaler.fit_transform(Xf_tr)
        Xf_te_s  = scaler.transform(Xf_te)

        Xf_aug, y_aug = augment(Xf_tr_s, y_fold)

        if len(np.unique(y_aug)) < 2:
            only_class = int(y_aug[0])
            val_true.extend(y_test.tolist())
            val_pred.extend([only_class] * len(y_test))
            val_prob.extend([float(only_class)] * len(y_test))
            continue

        clf = make_clf(clf_name)
        clf.fit(Xf_aug, y_aug)
        preds = clf.predict(Xf_te_s)
        proba = clf.predict_proba(Xf_te_s)

        val_true.extend(y_test.tolist())
        val_pred.extend(preds.tolist())
        val_prob.extend(
            proba[:, 1].tolist() if proba.shape[1] == 2
            else [float(clf.classes_[0])] * len(y_test)
        )

    acc, sens, spec, auc, prec, f1 = metrics(val_true, val_pred, val_prob)
    return val_true, val_pred, val_prob, acc, sens, spec, auc, f1


# ── Run one combination ───────────────────────────────────────────────────────

def run_combination(label, feat_idx, feature_names, X_tr, y_tr, X_te, y_te,
                    clf_name, df_train, df_test):
    print(f"\n  [{label}]")

    vt, vp, vprob, tr_acc, val_acc, val_sens, val_spec, val_auc, val_f1 = \
        run_loocv(X_tr, y_tr, feat_idx, clf_name)

    # Per-patient LOOCV detail
    for p, yt, yp in zip(df_train["patient"], vt, vp):
        ok = "OK   " if yt == yp else "WRONG"
        print(f"    {ok}  {p:35s}  true={'CANCER' if yt else 'HEALTHY':7s}  "
              f"pred={'CANCER' if yp else 'HEALTHY'}")

    auc_str = f"{val_auc:.3f}" if not np.isnan(val_auc) else "N/A "
    f1_str  = f"{val_f1:.3f}"  if not np.isnan(val_f1)  else "N/A"
    print(f"    LOOCV -> Acc={val_acc:.0%}  Recall={val_sens:.0%}  "
          f"Spec={val_spec:.0%}  AUC={auc_str}  F1={f1_str}")

    # Final model on all train → test
    scaler_f = StandardScaler()
    Xf_tr_s  = scaler_f.fit_transform(X_tr[:, feat_idx])
    Xf_te_s  = scaler_f.transform(X_te[:, feat_idx])
    Xf_aug, y_aug = augment(Xf_tr_s, y_tr)
    clf_f = make_clf(clf_name)
    clf_f.fit(Xf_aug, y_aug)

    te_pred = clf_f.predict(Xf_te_s)
    te_prob = clf_f.predict_proba(Xf_te_s)[:, 1]
    _, te_sens, te_spec, te_auc, te_prec, te_f1 = metrics(y_te, te_pred, te_prob)

    print(f"    Test  -> Recall={te_sens:.0%}  F1={te_f1:.3f}  "
          f"({'N/A — no healthy patients in test' if np.isnan(te_spec) else f'Spec={te_spec:.0%}'})")

    return {
        "Combination": label,
        "Features": [feature_names[i] for i in feat_idx],
        "Train acc": tr_acc,
        "LOOCV acc": val_acc,
        "LOOCV sens": val_sens,
        "LOOCV spec": val_spec,
        "LOOCV AUC": val_auc,
        "LOOCV F1": val_f1,
        "Test sens": te_sens,
        "Test F1": te_f1,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    feat_cols = [c for c in df.columns if c not in ("patient", "label", "split")]

    df_train = df[df["split"] == "TRAIN"].reset_index(drop=True)
    df_test  = df[df["split"] == "TEST"].reset_index(drop=True)

    train_idx = df[df["split"] == "TRAIN"].index.tolist()
    test_idx  = df[df["split"] == "TEST"].index.tolist()

    X_all = df[feat_cols].values.astype(float)
    y_tr  = df_train["label"].values.astype(int)
    y_te  = df_test["label"].values.astype(int)

    # ── Shared preprocessing (same for all combinations)
    X_clean, fnames = clean_features(X_all, list(feat_cols), train_idx)
    X_tr = X_clean[train_idx]
    X_te = X_clean[test_idx]

    cv_keep  = cv_filter(X_tr, fnames)
    icc_vals = compute_icc(df, fnames)

    n_cv_removed  = (~cv_keep).sum()
    n_icc_high    = (icc_vals >= 0.75).sum()
    print(f"Preprocessing: {len(fnames)} features | "
          f"CV removed={n_cv_removed} | ICC>=0.75: {n_icc_high}")

    # ── Feature indices for each selection method
    idx_varfs = select_varfs(X_tr, y_tr, fnames, icc_vals, cv_keep)
    idx_anova = select_anova(X_tr, y_tr, fnames, cv_keep)

    print(f"\nVaRFS features: {[fnames[i] for i in idx_varfs]}")
    print(f"ANOVA features: {[fnames[i] for i in idx_anova]}")

    # ── Run all 4 combinations
    print("\n" + "=" * 65)
    print("COMPARISON: VaRFS vs ANOVA  x  RandomForest vs SVM")
    print("=" * 65)

    combinations = [
        ("VaRFS + RandomForest",      idx_varfs, "RandomForest"),
        ("VaRFS + SVM",               idx_varfs, "SVM"),
        ("VaRFS + LogisticRegr",      idx_varfs, "LogisticRegression"),
        ("VaRFS + KNN",               idx_varfs, "KNN"),
        ("VaRFS + NaiveBayes",        idx_varfs, "NaiveBayes"),
        ("VaRFS + DecisionTree",      idx_varfs, "DecisionTree"),
        ("ANOVA + RandomForest",      idx_anova, "RandomForest"),
        ("ANOVA + SVM",               idx_anova, "SVM"),
        ("ANOVA + LogisticRegr",      idx_anova, "LogisticRegression"),
        ("ANOVA + KNN",               idx_anova, "KNN"),
        ("ANOVA + NaiveBayes",        idx_anova, "NaiveBayes"),
        ("ANOVA + DecisionTree",      idx_anova, "DecisionTree"),
    ]

    results = []
    for label, feat_idx, clf_name in combinations:
        r = run_combination(label, feat_idx, fnames,
                            X_tr, y_tr, X_te, y_te,
                            clf_name, df_train, df_test)
        results.append(r)

    # ── Summary table
    print("\n" + "=" * 65)
    print("SUMMARY TABLE")
    print("=" * 65)
    print(f"  {'Combination':<25} {'Train':>6} {'LOO-Acc':>8} "
          f"{'LOO-Recall':>11} {'LOO-Spec':>9} {'LOO-AUC':>8} {'LOO-F1':>7} {'Test-Recall':>12}")
    print("  " + "-" * 90)
    for r in results:
        auc_s = f"{r['LOOCV AUC']:.3f}" if not np.isnan(r["LOOCV AUC"]) else " N/A"
        f1_s  = f"{r['LOOCV F1']:.3f}"  if not np.isnan(r["LOOCV F1"])  else " N/A"
        print(f"  {r['Combination']:<25} {r['Train acc']:>5.0%}  "
              f"{r['LOOCV acc']:>7.0%}  {r['LOOCV sens']:>10.0%}  "
              f"{r['LOOCV spec']:>8.0%}  {auc_s:>7}  {f1_s:>6}  {r['Test sens']:>11.0%}")

    # ── Best combination
    best = max(results, key=lambda r: (r["Test sens"], r["LOOCV acc"], r["LOOCV AUC"]))
    print(f"\n  Best: {best['Combination']}")
    print(f"  Features: {best['Features']}")
