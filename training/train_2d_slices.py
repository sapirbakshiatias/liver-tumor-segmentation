"""
Patient-level LOOCV on 2D per-slice features.

Protocol:
  - Train on ALL slices from 4 patients
  - Test  on ALL slices from 1 patient
  - Patient prediction = weighted average of per-slice probabilities
    (weight = liver_px / max_liver_px_of_patient — more liver = more reliable slice)
  - Compare to 3D series-level results

Run: .venv\Scripts\python.exe train_2d_slices.py
"""

import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler

CSV_PATH = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\slice_radiomics_2d.csv"

df = pd.read_csv(CSV_PATH)

# Drop non-finite rows
feat_cols = [c for c in df.columns
             if c not in ("series","patient","group","label","slice_idx","liver_px")]
df = df.dropna(subset=feat_cols)
df = df[np.all(np.isfinite(df[feat_cols].values), axis=1)]

groups  = df["group"].tolist()
y_all   = df["label"].values.astype(int)
X_all   = df[feat_cols].values.astype(float)
PATIENTS = list(dict.fromkeys(groups))   # preserve order

# Per-patient max liver_px (for slice weights)
max_px = df.groupby("group")["liver_px"].max().to_dict()


def augment(X, y):
    n_min = (y == 0).sum()
    if n_min >= 2:
        return SMOTE(k_neighbors=min(1, n_min-1), random_state=42).fit_resample(X, y)
    if n_min == 1:
        return RandomOverSampler(random_state=42).fit_resample(X, y)
    return X, y


def metrics(y_true, y_pred, y_prob=None):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = int(((y_true==1)&(y_pred==1)).sum())
    tn = int(((y_true==0)&(y_pred==0)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())
    acc  = (tp+tn)/len(y_true)
    sens = tp/(tp+fn) if tp+fn>0 else float("nan")
    spec = tn/(tn+fp) if tn+fp>0 else float("nan")
    prec = tp/(tp+fp) if tp+fp>0 else float("nan")
    f1   = (2*prec*sens/(prec+sens)
            if not any(np.isnan([prec,sens])) and prec+sens>0 else float("nan"))
    auc  = float("nan")
    if y_prob is not None and len(np.unique(y_true))==2:
        try: auc = roc_auc_score(y_true, y_prob)
        except: pass
    return acc, sens, spec, auc, f1


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


def loocv_2d(clf_name):
    fold_true, fold_pred, fold_prob = [], [], []
    per_patient = {}

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

        # Weight by liver_px (normalised per patient)
        lx       = df[test_mask]["liver_px"].values.astype(float)
        weights  = lx / lx.max()
        avg_prob = float(np.average(slice_probs, weights=weights))
        pred     = int(avg_prob >= 0.5)

        fold_true.append(true_label)
        fold_pred.append(pred)
        fold_prob.append(avg_prob)
        per_patient[patient] = (true_label, pred, avg_prob,
                                int(test_mask.sum()), float(slice_probs.mean()),
                                float(slice_probs.std()))

    acc, sens, spec, auc, f1 = metrics(fold_true, fold_pred, fold_prob)
    return acc, sens, spec, auc, f1, per_patient


# ── Main ──────────────────────────────────────────────────────────────────────

print(f"Dataset: {len(df)} slices | {len(PATIENTS)} patients | {len(feat_cols)} features")
print(f"Slices per patient: " +
      "  ".join(f"{p.split('_')[1]}={sum(g==p for g in groups)}" for p in PATIENTS))

CLF_NAMES = ["KNN", "RandomForest", "GradientBoost", "SVM", "MLP"]

print(f"\n{'='*120}")
print("2D SLICE-LEVEL LOOCV  (train on slices of 4 patients, test on slices of 1 patient)")
print(f"{'='*120}")
print(f"  {'Model':<18} {'Acc':>5} {'Sens':>5} {'Spec':>5} {'AUC':>6} {'F1':>6} | "
      f"{'Patient_1':>13} {'Patient_2':>13} {'Patient_KB':>13} {'Patient_GA':>13} {'Patient_VT':>13}")
print(f"  {'-'*118}")

results = []
for clf_name in CLF_NAMES:
    acc, sens, spec, auc, f1, pp = loocv_2d(clf_name)
    auc_s = f"{auc:.3f}" if auc==auc else "N/A"
    f1_s  = f"{f1:.3f}"  if f1==f1   else "N/A"

    def cell(p):
        tl, pl, prob, n_sl, mean_sl, std_sl = pp[p]
        ok = "OK" if tl==pl else ("FP" if pl==1 and tl==0 else "FN")
        return f"{ok}({prob:.2f})"

    wrong = [p for p in PATIENTS if pp[p][0] != pp[p][1]]
    wrong_s = ""
    if wrong:
        types = [("FP" if pp[p][1]==1 and pp[p][0]==0 else "FN") for p in wrong]
        wrong_s = "  <- " + ", ".join(f"{p}({t})" for p, t in zip(wrong, types))

    print(f"  {clf_name:<18} {acc:>4.0%} {sens:>5.0%} {spec:>5.0%} {auc_s:>6} {f1_s:>6} | "
          f"{cell('Patient_1'):>13} {cell('Patient_2'):>13} "
          f"{cell('Patient_KB'):>13} {cell('Patient_GA'):>13} "
          f"{cell('Patient_VT'):>13}{wrong_s}")
    results.append((clf_name, acc, sens, spec, auc, f1, pp))

# ── Detailed per-patient breakdown for best model ─────────────────────────────
best = max(results, key=lambda r: (r[1], r[5] if r[5]==r[5] else 0))
print(f"\nBest model: {best[0]}  (Acc={best[1]:.0%}  F1={best[5]:.3f})")
print(f"\nDetailed per-patient breakdown ({best[0]}):")
print(f"  {'Patient':<15} {'True':>8} {'Pred':>8} {'Prob':>7}  {'Slices':>7}  "
      f"{'SliceMean':>10}  {'SliceStd':>9}")
print(f"  {'-'*70}")
for p in PATIENTS:
    tl, pl, prob, n_sl, mean_sl, std_sl = best[6][p]
    true_s = "Cancer"  if tl==1 else "Healthy"
    pred_s = "Cancer"  if pl==1 else "Healthy"
    ok     = "OK" if tl==pl else "WRONG"
    print(f"  {p:<15} {true_s:>8} {pred_s:>8} {prob:>7.3f}  {n_sl:>7}  "
          f"{mean_sl:>10.3f}  {std_sl:>9.3f}  {ok}")

# ── Comparison with 3D results ────────────────────────────────────────────────
print(f"\n{'='*60}")
print("COMPARISON: 3D series-level vs 2D slice-level (VaRFS+KNN)")
print(f"{'='*60}")
print("  3D (series, 5 features, weighted 1/rank):  Acc=100%  VT=OK(0.39)")
best_2d = next(r for r in results if r[0]=="KNN")
knn_vt  = best_2d[6]["Patient_VT"]
print(f"  2D (slices, 16 features, weighted liver):  "
      f"Acc={best_2d[1]:.0%}  VT={'OK' if knn_vt[1]==0 else 'WRONG'}({knn_vt[2]:.2f})")
