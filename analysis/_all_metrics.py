"""
Full metrics table for every model combination:
Accuracy, Precision, Recall (Sensitivity), Specificity, F1, AUC
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")

from train_all_series_report import (
    clean_features, cv_filter, compute_icc, select_varfs, select_anova,
    augment, make_clf, metrics, _series_weight,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

CSV_PATH = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\all_series_radiomics.csv"
CSV_2D   = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\slice_radiomics_2d.csv"

# ── helpers ───────────────────────────────────────────────────────────────────

def full_metrics(y_true, y_pred, y_prob=None):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = int(((y_true==1)&(y_pred==1)).sum())
    tn = int(((y_true==0)&(y_pred==0)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())
    acc  = (tp+tn)/len(y_true)
    sens = tp/(tp+fn)   if (tp+fn)>0 else float("nan")   # recall
    spec = tn/(tn+fp)   if (tn+fp)>0 else float("nan")
    prec = tp/(tp+fp)   if (tp+fp)>0 else float("nan")
    f1   = (2*prec*sens/(prec+sens)
            if not any(np.isnan([prec,sens])) and prec+sens>0 else float("nan"))
    auc  = float("nan")
    if y_prob is not None and len(np.unique(y_true))==2:
        try: auc = roc_auc_score(y_true, y_prob)
        except: pass
    return acc, prec, sens, spec, f1, auc, tp, tn, fp, fn

def fmt(v):
    return f"{v:.1%}" if v==v else "  N/A "

def fmt3(v):
    return f"{v:.3f}" if v==v else "  N/A"

SEP  = "="*130
SEP2 = "-"*130
HDR  = (f"  {'Model':<32} {'Acc':>6} {'Prec':>6} {'Recall':>7} {'Spec':>6} "
        f"{'F1':>6} {'AUC':>6}   TP TN FP FN  | Errors")

# ══════════════════════════════════════════════════════════════════════════════
# 3D RADIOMICS
# ══════════════════════════════════════════════════════════════════════════════

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

PATIENTS = ["Patient_1","Patient_2","Patient_KB","Patient_GA","Patient_VT"]
CLF_NAMES = ["KNN","RandomForest","GradientBoost","SVM",
             "LogisticRegr","NaiveBayes","DecisionTree","MLP"]

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
            test_series = [series_names[i] for i,m in enumerate(test_mask) if m]
            w = np.array([_series_weight(s) for s in test_series])
            avg_prob = float(np.average(probs, weights=w))
        else:
            avg_prob = float(probs.mean())
        pred = int(avg_prob>=0.5)
        fold_true.append(true_label); fold_pred.append(pred); fold_prob.append(avg_prob)
        pp[patient] = (true_label, pred, avg_prob)
    m = full_metrics(fold_true, fold_pred, fold_prob)
    return m, pp

def row3d(label, feat_idx, clf_name, weighted):
    m, pp = loocv_3d(feat_idx, clf_name, weighted)
    acc,prec,rec,spec,f1,auc,tp,tn,fp,fn = m
    wrong = [p for p in PATIENTS if pp[p][0]!=pp[p][1]]
    errs = ", ".join(
        f"{p}({'FP' if pp[p][1]==1 and pp[p][0]==0 else 'FN'})"
        for p in wrong) if wrong else "-"
    name = f"{label}+{clf_name}"
    print(f"  {name:<32} {fmt(acc):>6} {fmt(prec):>6} {fmt(rec):>7} {fmt(spec):>6} "
          f"{fmt3(f1):>6} {fmt3(auc):>6}   {tp:2d} {tn:2d} {fp:2d} {fn:2d}  | {errs}")

print(SEP)
print("FULL METRICS TABLE -- All model combinations")
print("Ground truth: P1=Cancer  P2=Cancer  KB=Cancer  GA=Healthy  VT=Healthy")
print("Recall=Sensitivity=TPR | Spec=Specificity=TNR | Prec=Precision=PPV")
print(SEP)

# ── Stage 1 ───────────────────────────────────────────────────────────────────
print(f"\n'-'*130")
print("  STAGE 1 | 3D Radiomics | Simple average (no weighting)")
print(SEP2)
print(HDR)
print(SEP2)
for feat_label, feat_idx in [("VaRFS", idx_varfs), ("ANOVA", idx_anova)]:
    for clf in CLF_NAMES:
        row3d(feat_label, feat_idx, clf, weighted=False)

# ── Stage 2 ───────────────────────────────────────────────────────────────────
print(f"\n'-'*130")
print("  STAGE 2 | 3D Radiomics | Weighted 1/rank")
print(SEP2)
print(HDR)
print(SEP2)
for feat_label, feat_idx in [("VaRFS", idx_varfs), ("ANOVA", idx_anova)]:
    for clf in CLF_NAMES:
        row3d(feat_label, feat_idx, clf, weighted=True)

# ── Attention (hardcoded from saved run) ──────────────────────────────────────
print(f"\n'-'*130")
print("  STAGE 2 | 3D Radiomics | Gated Attention MLP")
print(SEP2)
print(HDR)
print(SEP2)
attn = [
    ("VaRFS+Attention",  0.80, 1.00, 0.67, 1.00, 0.800, 1.000, 2,2,0,1, "Patient_1(FN)"),
    ("ANOVA+Attention",  0.80, 0.75, 1.00, 0.50, 0.857, 0.667, 3,1,1,0, "Patient_VT(FP)"),
]
for name,acc,prec,rec,spec,f1,auc,tp,tn,fp,fn,err in attn:
    print(f"  {name:<32} {fmt(acc):>6} {fmt(prec):>6} {fmt(rec):>7} {fmt(spec):>6} "
          f"{fmt3(f1):>6} {fmt3(auc):>6}   {tp:2d} {tn:2d} {fp:2d} {fn:2d}  | {err}")

# ══════════════════════════════════════════════════════════════════════════════
# 2D SLICE RADIOMICS
# ══════════════════════════════════════════════════════════════════════════════

df2 = pd.read_csv(CSV_2D)
feat_cols2 = [c for c in df2.columns
              if c not in ("series","patient","group","label","slice_idx","liver_px")]
df2 = df2.dropna(subset=feat_cols2)
df2 = df2[np.all(np.isfinite(df2[feat_cols2].values), axis=1)]
groups2 = df2["group"].tolist()
y2      = df2["label"].values.astype(int)
X2      = df2[feat_cols2].values.astype(float)
max_px  = df2.groupby("group")["liver_px"].max().to_dict()
PATIENTS2 = list(dict.fromkeys(groups2))

from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

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
    m = full_metrics(fold_true, fold_pred, fold_prob)
    return m, pp

print(f"\n'-'*130")
print("  STAGE 3 | 2D Slice Radiomics | Weighted by liver_px (795 slices)")
print(SEP2)
print(HDR)
print(SEP2)
for clf in ["KNN","RandomForest","GradientBoost","SVM","MLP"]:
    m, pp = loocv_2d(clf)
    acc,prec,rec,spec,f1,auc,tp,tn,fp,fn = m
    wrong = [p for p in PATIENTS2 if pp[p][0]!=pp[p][1]]
    errs = ", ".join(
        f"{p}({'FP' if pp[p][1]==1 and pp[p][0]==0 else 'FN'})"
        for p in wrong) if wrong else "-"
    name = f"2D-Slice+{clf}"
    print(f"  {name:<32} {fmt(acc):>6} {fmt(prec):>6} {fmt(rec):>7} {fmt(spec):>6} "
          f"{fmt3(f1):>6} {fmt3(auc):>6}   {tp:2d} {tn:2d} {fp:2d} {fn:2d}  | {errs}")

# ══════════════════════════════════════════════════════════════════════════════
# 2D CNN (hardcoded from 5562s run)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n'-'*130")
print("  STAGE 4 | 2D CNN (ResNet18 + SE-Attention + Grad-CAM, run on CPU)")
print(SEP2)
print(HDR)
print(SEP2)
cnn = [
    ("2D-CNN+SimpleCNN",         0.60, 0.60, 1.00, 0.00, 0.750, 0.333, 3,0,2,0, "GA(FP), VT(FP)"),
    ("2D-CNN+ResNet18-Linear",   0.60, 0.60, 1.00, 0.00, 0.750, 0.000, 3,0,2,0, "GA(FP), VT(FP)"),
    ("2D-CNN+ResNet18-SE-Attn",  0.60, 0.60, 1.00, 0.00, 0.750, 0.000, 3,0,2,0, "GA(FP), VT(FP)"),
    ("2D-CNN+ResNet18-FineTune", 0.60, 0.60, 1.00, 0.00, 0.750, 0.000, 3,0,2,0, "GA(FP), VT(FP)"),
]
for name,acc,prec,rec,spec,f1,auc,tp,tn,fp,fn,err in cnn:
    print(f"  {name:<32} {fmt(acc):>6} {fmt(prec):>6} {fmt(rec):>7} {fmt(spec):>6} "
          f"{fmt3(f1):>6} {fmt3(auc):>6}   {tp:2d} {tn:2d} {fp:2d} {fn:2d}  | {err}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  BEST RESULTS SUMMARY")
print(SEP)
print(f"  {'Model':<35} {'Acc':>6} {'Prec':>6} {'Recall':>7} {'Spec':>6} {'F1':>6} {'AUC':>6}")
print(SEP2)
bests = [
    ("VaRFS+KNN  (3D, 1/rank)",    1.00,1.00,1.00,1.00,1.000,1.000),
    ("VaRFS+Attention (3D,1/rank)",0.80,1.00,0.67,1.00,0.800,1.000),
    ("VaRFS+RF   (3D, 1/rank)",    0.80,0.75,1.00,0.50,0.857,0.833),
    ("ANOVA+RF   (3D, 1/rank)",    0.80,0.75,1.00,0.50,0.857,0.667),
    ("2D-Slice+RF",                0.80,0.75,1.00,0.50,0.857,0.833),
    ("2D-Slice+SVM",               0.80,0.75,1.00,0.50,0.857,0.833),
    ("2D-CNN (all variants)",      0.60,0.60,1.00,0.00,0.750,0.167),
]
for name,acc,prec,rec,spec,f1,auc in bests:
    print(f"  {name:<35} {fmt(acc):>6} {fmt(prec):>6} {fmt(rec):>7} {fmt(spec):>6} "
          f"{fmt3(f1):>6} {fmt3(auc):>6}")

print(f"\n  Legend:")
print(f"  Acc    = Accuracy        = (TP+TN)/(TP+TN+FP+FN)")
print(f"  Prec   = Precision       = TP/(TP+FP)  -- of all Cancer predictions, how many correct?")
print(f"  Recall = Sensitivity/TPR = TP/(TP+FN)  -- of all Cancer patients, how many detected?")
print(f"  Spec   = Specificity/TNR = TN/(TN+FP)  -- of all Healthy patients, how many detected?")
print(f"  F1     = 2*Prec*Recall/(Prec+Recall)   -- harmonic mean of Precision and Recall")
print(f"  AUC    = Area Under ROC Curve           -- 1.0=perfect, 0.5=random")
print(f"  TP/TN/FP/FN:")
print(f"    TP = Cancer classified correctly  | TN = Healthy classified correctly")
print(f"    FP = Healthy classified as Cancer (unnecessary biopsy)")
print(f"    FN = Cancer classified as Healthy (missed diagnosis)")
print(SEP)
