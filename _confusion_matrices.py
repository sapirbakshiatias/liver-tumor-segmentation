"""
Confusion matrix for every model combination.
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from train_all_series_report import (
    clean_features, cv_filter, compute_icc, select_varfs, select_anova,
    augment, make_clf, _series_weight,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

CSV_PATH = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\all_series_radiomics.csv"
CSV_2D   = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\slice_radiomics_2d.csv"

PATIENTS = ["Patient_1","Patient_2","Patient_KB","Patient_GA","Patient_VT"]

# ── Load 3D data ──────────────────────────────────────────────────────────────
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

# ── Load 2D data ──────────────────────────────────────────────────────────────
df2 = pd.read_csv(CSV_2D)
feat_cols2 = [c for c in df2.columns
              if c not in ("series","patient","group","label","slice_idx","liver_px")]
df2 = df2.dropna(subset=feat_cols2)
df2 = df2[np.all(np.isfinite(df2[feat_cols2].values), axis=1)]
groups2   = df2["group"].tolist()
y2        = df2["label"].values.astype(int)
X2        = df2[feat_cols2].values.astype(float)
PATIENTS2 = list(dict.fromkeys(groups2))


# ── Confusion matrix printer ──────────────────────────────────────────────────

def confusion_matrix_print(title, y_true, y_pred, y_prob, per_patient):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = int(((y_true==1)&(y_pred==1)).sum())
    tn = int(((y_true==0)&(y_pred==0)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())

    acc  = (tp+tn)/len(y_true)
    prec = tp/(tp+fp)   if (tp+fp)>0 else float("nan")
    rec  = tp/(tp+fn)   if (tp+fn)>0 else float("nan")
    spec = tn/(tn+fp)   if (tn+fp)>0 else float("nan")
    f1   = (2*prec*rec/(prec+rec)
            if not any(np.isnan([prec,rec])) and prec+rec>0 else float("nan"))
    auc  = float("nan")
    if len(np.unique(y_true))==2:
        try: auc = roc_auc_score(y_true, y_prob)
        except: pass

    W = 54
    border = "+" + "-"*W + "+"

    def pf(v): return f"{v:.1%}" if v==v else "N/A"
    def p3(v): return f"{v:.3f}" if v==v else "N/A "

    print(border)
    t = f"  {title}"
    print(f"|{t:<{W}}|")
    print(border)
    print(f"|{'':^{W}}|")
    print(f"|{'Predicted:  Cancer        Healthy':^{W}}|")
    print(f"|  Actual Cancer    {tp:2d} (TP)        {fn:2d} (FN)   |")
    print(f"|  Actual Healthy   {fp:2d} (FP)        {tn:2d} (TN)   |")
    print(f"|{'':^{W}}|")
    print(f"|  Acc={pf(acc)}  Prec={pf(prec)}  Recall={pf(rec):<7}|")
    print(f"|  Spec={pf(spec)}  F1={p3(f1)}  AUC={p3(auc):<9}|")
    print(f"|{'':^{W}}|")

    # Per-patient detail
    pats_str = []
    for pat, (tl, pl, prob) in per_patient.items():
        short = pat.replace("Patient_","P")
        status = "OK" if tl==pl else ("FP" if pl==1 and tl==0 else "FN")
        pats_str.append(f"{short}:{status}({prob:.2f})")
    line = "  " + "  ".join(pats_str)
    print(f"|{line:<{W}}|")
    print(border)
    print()


# ── 3D LOOCV helper ───────────────────────────────────────────────────────────

def loocv_3d(feat_idx, clf_name, weighted):
    fold_true, fold_pred, fold_prob, pp = [], [], [], {}
    for patient in PATIENTS:
        test_mask  = np.array([g==patient for g in groups])
        train_mask = ~test_mask
        Xf_tr = X_clean[train_mask][:,feat_idx]
        Xf_te = X_clean[test_mask][:,feat_idx]
        y_tr  = y_all[train_mask]
        true_label = int(np.round(y_all[test_mask].mean()))
        sc = StandardScaler()
        Xf_tr_s = sc.fit_transform(Xf_tr)
        Xf_te_s = sc.transform(Xf_te)
        Xf_aug, y_aug = augment(Xf_tr_s, y_tr)
        clf = make_clf(clf_name)
        clf.fit(Xf_aug, y_aug)
        probs = clf.predict_proba(Xf_te_s)[:,1]
        if weighted:
            ts = [series_names[i] for i,m in enumerate(test_mask) if m]
            w  = np.array([_series_weight(s) for s in ts])
            avg_prob = float(np.average(probs, weights=w))
        else:
            avg_prob = float(probs.mean())
        pred = int(avg_prob>=0.5)
        fold_true.append(true_label)
        fold_pred.append(pred)
        fold_prob.append(avg_prob)
        pp[patient] = (true_label, pred, avg_prob)
    return fold_true, fold_pred, fold_prob, pp


# ── 2D LOOCV helper ───────────────────────────────────────────────────────────

def make_clf2(name):
    return {
        "KNN":           KNeighborsClassifier(n_neighbors=5),
        "RandomForest":  RandomForestClassifier(n_estimators=200,
                             class_weight="balanced", random_state=42),
        "GradientBoost": GradientBoostingClassifier(n_estimators=100,
                             max_depth=3, learning_rate=0.1, random_state=42),
        "SVM":           SVC(kernel="rbf", C=1.0, class_weight="balanced",
                             probability=True, random_state=42),
        "MLP":           MLPClassifier(hidden_layer_sizes=(64,32),
                             max_iter=500, random_state=42,
                             early_stopping=True, validation_fraction=0.1),
    }[name]

def augment2(X, y):
    n_min = (y==0).sum()
    if n_min>=2: return SMOTE(k_neighbors=min(1,n_min-1), random_state=42).fit_resample(X,y)
    if n_min==1: return RandomOverSampler(random_state=42).fit_resample(X,y)
    return X, y

def loocv_2d(clf_name):
    fold_true, fold_pred, fold_prob, pp = [], [], [], {}
    for patient in PATIENTS2:
        test_mask  = np.array([g==patient for g in groups2])
        train_mask = ~test_mask
        X_tr = X2[train_mask]; X_te = X2[test_mask]
        y_tr = y2[train_mask]
        true_label = int(y2[test_mask][0])
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
        X_aug, y_aug = augment2(X_tr_s, y_tr)
        clf = make_clf2(clf_name)
        clf.fit(X_aug, y_aug)
        slice_probs = clf.predict_proba(X_te_s)[:,1]
        lx = df2[test_mask]["liver_px"].values.astype(float)
        avg_prob = float(np.average(slice_probs, weights=lx/lx.max()))
        pred = int(avg_prob>=0.5)
        fold_true.append(true_label); fold_pred.append(pred); fold_prob.append(avg_prob)
        pp[patient] = (true_label, pred, avg_prob)
    return fold_true, fold_pred, fold_prob, pp


# ══════════════════════════════════════════════════════════════════════════════
# PRINT ALL
# ══════════════════════════════════════════════════════════════════════════════

CLF_NAMES = ["KNN","RandomForest","GradientBoost","SVM","LogisticRegr","NaiveBayes","DecisionTree","MLP"]

print("=" * 56)
print("  CONFUSION MATRICES -- All model combinations")
print("  5 patients | Patient-level LOOCV")
print("  TP/FN = Cancer | FP/TN = Healthy")
print("=" * 56)
print()

# ── Stage 1: Simple average ───────────────────────────────────────────────────
print(">>> STAGE 1: 3D Radiomics | Simple average (no weighting)")
print()
for feat_label, feat_idx in [("VaRFS", idx_varfs), ("ANOVA", idx_anova)]:
    for clf in CLF_NAMES:
        yt, yp, yprob, pp = loocv_3d(feat_idx, clf, weighted=False)
        confusion_matrix_print(
            f"{feat_label} + {clf} | 3D simple-avg",
            yt, yp, yprob, pp)

# ── Stage 2: Weighted ─────────────────────────────────────────────────────────
print(">>> STAGE 2: 3D Radiomics | Weighted 1/rank")
print()
for feat_label, feat_idx in [("VaRFS", idx_varfs), ("ANOVA", idx_anova)]:
    for clf in CLF_NAMES:
        yt, yp, yprob, pp = loocv_3d(feat_idx, clf, weighted=True)
        confusion_matrix_print(
            f"{feat_label} + {clf} | 3D weighted-1/rank",
            yt, yp, yprob, pp)

# ── Attention (hardcoded) ─────────────────────────────────────────────────────
print(">>> STAGE 2: 3D Radiomics | Gated Attention MLP")
print()
attn_results = [
    ("VaRFS + Attention | 3D weighted-1/rank",
     [1,1,1,0,0],[1,0,1,0,0],[0.61,0.45,0.94,0.11,0.41],
     {"Patient_1":(1,1,0.61),"Patient_2":(1,0,0.45),
      "Patient_KB":(1,1,0.94),"Patient_GA":(0,0,0.11),"Patient_VT":(0,0,0.41)}),
    ("ANOVA + Attention | 3D weighted-1/rank",
     [1,1,1,0,0],[1,1,1,0,1],[0.61,0.91,0.56,0.29,0.84],
     {"Patient_1":(1,1,0.61),"Patient_2":(1,1,0.91),
      "Patient_KB":(1,1,0.56),"Patient_GA":(0,0,0.29),"Patient_VT":(0,1,0.84)}),
]
for title, yt, yp, yprob, pp in attn_results:
    confusion_matrix_print(title, yt, yp, yprob, pp)

# ── Stage 3: 2D Slice ─────────────────────────────────────────────────────────
print(">>> STAGE 3: 2D Slice Radiomics | Weighted by liver_px")
print()
for clf in ["KNN","RandomForest","GradientBoost","SVM","MLP"]:
    yt, yp, yprob, pp = loocv_2d(clf)
    confusion_matrix_print(
        f"2D-Slice + {clf} | weighted liver_px",
        yt, yp, yprob, pp)

# ── Stage 4: 2D CNN (hardcoded) ───────────────────────────────────────────────
print(">>> STAGE 4: 2D CNN (ResNet18 variants)")
print()
cnn_results = [
    ("SimpleCNN | 2D raw CT slices",
     [1,1,1,0,0],[1,1,1,1,1],[0.963,0.786,0.889,0.941,0.961],
     {"Patient_1":(1,1,0.963),"Patient_2":(1,1,0.786),
      "Patient_KB":(1,1,0.889),"Patient_GA":(0,1,0.941),"Patient_VT":(0,1,0.961)}),
    ("ResNet18-Linear | 2D raw CT slices",
     [1,1,1,0,0],[1,1,1,1,1],[0.665,0.710,0.567,0.751,0.863],
     {"Patient_1":(1,1,0.665),"Patient_2":(1,1,0.710),
      "Patient_KB":(1,1,0.567),"Patient_GA":(0,1,0.751),"Patient_VT":(0,1,0.863)}),
    ("ResNet18-SE-Attention | 2D raw CT slices",
     [1,1,1,0,0],[1,1,1,1,1],[0.780,0.887,0.630,0.903,0.974],
     {"Patient_1":(1,1,0.780),"Patient_2":(1,1,0.887),
      "Patient_KB":(1,1,0.630),"Patient_GA":(0,1,0.903),"Patient_VT":(0,1,0.974)}),
    ("ResNet18-FineTune | 2D raw CT slices",
     [1,1,1,0,0],[1,1,1,1,1],[0.742,0.815,0.869,0.976,0.928],
     {"Patient_1":(1,1,0.742),"Patient_2":(1,1,0.815),
      "Patient_KB":(1,1,0.869),"Patient_GA":(0,1,0.976),"Patient_VT":(0,1,0.928)}),
]
for title, yt, yp, yprob, pp in cnn_results:
    confusion_matrix_print(title, yt, yp, yprob, pp)

print("=" * 56)
print("  LEGEND")
print("=" * 56)
print("  TP  Cancer   -> Cancer   (correct)")
print("  TN  Healthy  -> Healthy  (correct)")
print("  FP  Healthy  -> Cancer   (unnecessary biopsy)")
print("  FN  Cancer   -> Healthy  (missed diagnosis!)")
print()
print("  Acc    = (TP+TN) / all")
print("  Prec   = TP / (TP+FP)")
print("  Recall = TP / (TP+FN)  = Sensitivity")
print("  Spec   = TN / (TN+FP)  = Specificity")
print("  F1     = 2*Prec*Recall / (Prec+Recall)")
print("  AUC    = Area Under ROC  (1.0=perfect)")
print()
print("  Per-patient: P1/P2/KB=Cancer  GA/VT=Healthy")
print("  Prob shown = P(Cancer) for that patient")
print("=" * 56)
