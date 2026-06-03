"""
Full results summary: confusion matrix + Accuracy + F1 for every model combination.

Covers:
  Stage 1 — 3D Radiomics, simple average
  Stage 2 — 3D Radiomics, weighted 1/rank
  Stage 3 — Gated Attention MLP
  Stage 4 — 2D Slice Radiomics
  Stage 5 — 2D + 3D Combined features (ANOVA+MLP best result)
  Stage 6 — 2D CNN (SimpleCNN, ResNet18 variants)

Run: .venv\Scripts\python.exe analysis\all_results_summary.py
"""
import sys, warnings
sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler

from train_all_series_report import (
    clean_features, cv_filter, compute_icc, select_varfs, select_anova,
    augment, make_clf, metrics, _series_weight,
)

# ── Confusion matrix printer ──────────────────────────────────────────────────

def print_cm(title, y_true, y_pred, y_prob, per_patient):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = int(((y_true==1)&(y_pred==1)).sum())
    tn = int(((y_true==0)&(y_pred==0)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())

    acc  = (tp+tn)/len(y_true)
    prec = tp/(tp+fp) if tp+fp>0 else float("nan")
    rec  = tp/(tp+fn) if tp+fn>0 else float("nan")
    spec = tn/(tn+fp) if tn+fp>0 else float("nan")
    f1   = (2*prec*rec/(prec+rec)
            if not any(v!=v for v in [prec,rec]) and prec+rec>0 else float("nan"))
    auc  = float("nan")
    if len(set(y_true))==2:
        try: auc = roc_auc_score(y_true, y_prob)
        except: pass

    W = 54
    sep = "+" + "-"*W + "+"
    def pf(v): return f"{v:.1%}" if v==v else "N/A"
    def p3(v): return f"{v:.3f}" if v==v else "N/A"

    print(sep)
    print(f"|  {title:<{W-2}}|")
    print(sep)
    print(f"|{'':{W}}|")
    print(f"|{'Predicted:  Cancer        Healthy':^{W}}|")
    print(f"|  Actual Cancer    {tp:2d} (TP)        {fn:2d} (FN)   |")
    print(f"|  Actual Healthy   {fp:2d} (FP)        {tn:2d} (TN)   |")
    print(f"|{'':{W}}|")
    print(f"|  Acc={pf(acc)}  Prec={pf(prec)}  Recall={pf(rec):<7}|")
    print(f"|  Spec={pf(spec)}  F1={p3(f1)}  AUC={p3(auc):<9}|")
    print(f"|{'':{W}}|")

    pats = []
    for pat, (tl, pl, prob) in per_patient.items():
        short = pat.replace("Patient_","P")
        status = "OK" if tl==pl else ("FP" if pl==1 and tl==0 else "FN")
        pats.append(f"{short}:{status}({prob:.2f})")
    line = "  " + "  ".join(pats)
    print(f"|{line:<{W}}|")
    print(sep)
    print()


# ── Load 3D data ──────────────────────────────────────────────────────────────

CSV_3D = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\all_series_radiomics.csv"
df3 = pd.read_csv(CSV_3D)
feat_cols  = [c for c in df3.columns if c not in ("series","patient","group","label")]
groups3    = df3["group"].tolist()
series3    = df3["series"].tolist()
y3         = df3["label"].values.astype(int)
X3_raw     = df3[feat_cols].values.astype(float)
PATIENTS   = ["Patient_1","Patient_2","Patient_KB","Patient_GA","Patient_VT"]

X3, fnames3   = clean_features(X3_raw, list(feat_cols))
cv_keep3      = cv_filter(X3, fnames3)
icc3          = compute_icc(df3, fnames3)
idx_v, f3,_,_ = select_varfs(X3, y3, fnames3, icc3, cv_keep3)
idx_a, _      = select_anova(X3, y3, fnames3, cv_keep3)


def loocv_3d(feat_idx, clf_name, weighted):
    yt, yp, yprob, pp = [], [], [], {}
    for pat in PATIENTS:
        tm = np.array([g==pat for g in groups3]); trm = ~tm
        Xtr = X3[trm][:,feat_idx]; Xte = X3[tm][:,feat_idx]
        ytr = y3[trm]; tl  = int(np.round(y3[tm].mean()))
        sc  = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
        Xa, ya = augment(Xtr_s, ytr)
        clf = make_clf(clf_name); clf.fit(Xa, ya)
        probs = clf.predict_proba(Xte_s)[:,1]
        if weighted:
            ts = [series3[i] for i,m in enumerate(tm) if m]
            w  = np.array([_series_weight(s) for s in ts])
            ap = float(np.average(probs, weights=w))
        else:
            ap = float(probs.mean())
        pred = int(ap>=0.5)
        yt.append(tl); yp.append(pred); yprob.append(ap)
        pp[pat] = (tl, pred, ap)
    return yt, yp, yprob, pp


# ── Load 2D slice data ────────────────────────────────────────────────────────

CSV_2D = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\slice_radiomics_2d.csv"
df2d = pd.read_csv(CSV_2D)
fc2d = [c for c in df2d.columns if c not in ("series","patient","group","label","slice_idx","liver_px")]
df2d = df2d.dropna(subset=fc2d)
df2d = df2d[np.all(np.isfinite(df2d[fc2d].values),axis=1)]
g2d  = df2d["group"].tolist(); y2d = df2d["label"].values.astype(int)
X2d  = df2d[fc2d].values.astype(float)
P2d  = list(dict.fromkeys(g2d))


def aug2(X, y):
    n = (y==0).sum()
    if n>=2: return SMOTE(k_neighbors=min(1,n-1),random_state=42).fit_resample(X,y)
    if n==1: return RandomOverSampler(random_state=42).fit_resample(X,y)
    return X,y

def make2(name):
    return {"KNN": KNeighborsClassifier(n_neighbors=5),
            "RF":  RandomForestClassifier(200,class_weight="balanced",random_state=42),
            "GB":  GradientBoostingClassifier(n_estimators=100,max_depth=3,learning_rate=0.1,random_state=42),
            "SVM": SVC(kernel="rbf",C=1,class_weight="balanced",probability=True,random_state=42),
            "MLP": MLPClassifier((64,32),max_iter=500,random_state=42,
                                 early_stopping=True,validation_fraction=0.1)}[name]

def loocv_2d(clf_name):
    yt,yp,yprob,pp = [],[],[],{}
    for pat in P2d:
        tm = np.array([g==pat for g in g2d]); trm = ~tm
        Xtr=X2d[trm]; Xte=X2d[tm]; ytr=y2d[trm]; tl=int(y2d[tm][0])
        sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
        Xa,ya=aug2(Xtr_s,ytr)
        clf=make2(clf_name); clf.fit(Xa,ya)
        sp=clf.predict_proba(Xte_s)[:,1]
        lx=df2d[tm]["liver_px"].values.astype(float)
        ap=float(np.average(sp,weights=lx/lx.max()))
        pred=int(ap>=0.5)
        yt.append(tl);yp.append(pred);yprob.append(ap);pp[pat]=(tl,pred,ap)
    return yt,yp,yprob,pp


# ── Load 2D+3D data ───────────────────────────────────────────────────────────

CSV_23 = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\slice_2d_plus_3d.csv"
df23 = pd.read_csv(CSV_23)
fc23 = [c for c in df23.columns if c not in ("series","patient","group","label","slice_idx","liver_px")]
df23 = df23.dropna(subset=fc23)
df23 = df23[np.all(np.isfinite(df23[fc23].values),axis=1)]
g23  = df23["group"].tolist(); y23 = df23["label"].values.astype(int)
X23r = df23[fc23].values.astype(float)
P23  = list(dict.fromkeys(g23))

X23, fn23    = clean_features(X23r, list(fc23))
cv23         = cv_filter(X23, fn23)
icc23        = compute_icc(df23, fn23)
iv23,_,_,_   = select_varfs(X23, y23, fn23, icc23, cv23)
ia23,_       = select_anova(X23, y23, fn23, cv23)

def aug23(X, y):
    n=(y==0).sum()
    if n>=2: return SMOTE(k_neighbors=min(n-1,5),sampling_strategy=1.0,random_state=42).fit_resample(X,y)
    if n==1: return RandomOverSampler(random_state=42).fit_resample(X,y)
    return X,y

def loocv_23(feat_idx, clf_name, threshold=0.45):
    pat_probs={}
    for pat in P23:
        tm=np.array([g==pat for g in g23]); trm=~tm
        Xtr=X23[trm][:,feat_idx]; Xte=X23[tm][:,feat_idx]
        ytr=y23[trm]; tl=int(y23[tm][0])
        sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
        Xa,ya=aug23(Xtr_s,ytr)
        clf=make2(clf_name); clf.fit(Xa,ya)
        sp=clf.predict_proba(Xte_s)[:,1]
        lx=df23[tm]["liver_px"].values.astype(float)
        ts=df23[tm]["series"].values
        rw=np.array([1.0/int(s.split("_s")[-1]) for s in ts])
        ap=float(np.average(sp,weights=(lx/lx.max())*rw))
        pat_probs[pat]=(tl,ap)
    yt=[pat_probs[p][0] for p in P23]
    yprob=[pat_probs[p][1] for p in P23]
    yp=[int(pr>=threshold) for pr in yprob]
    pp={p:(pat_probs[p][0],int(pat_probs[p][1]>=threshold),pat_probs[p][1]) for p in P23}
    return yt,yp,yprob,pp


# ══════════════════════════════════════════════════════════════════════════════
# PRINT ALL
# ══════════════════════════════════════════════════════════════════════════════

CLF3 = ["KNN","RandomForest","GradientBoost","SVM","LogisticRegr","NaiveBayes","DecisionTree","MLP"]
CLF2 = ["KNN","RF","GB","SVM","MLP"]

# ── Stage 1 ───────────────────────────────────────────────────────────────────
print("=" * 56)
print("  STAGE 1 | 3D Radiomics | Simple average (no weighting)")
print("=" * 56 + "\n")
for fl, fi in [("VaRFS", idx_v), ("ANOVA", idx_a)]:
    for clf in CLF3:
        yt,yp,yprob,pp = loocv_3d(fi, clf, weighted=False)
        print_cm(f"{fl} + {clf} | 3D simple-avg", yt, yp, yprob, pp)

# ── Stage 2 ───────────────────────────────────────────────────────────────────
print("=" * 56)
print("  STAGE 2 | 3D Radiomics | Weighted 1/rank")
print("=" * 56 + "\n")
for fl, fi in [("VaRFS", idx_v), ("ANOVA", idx_a)]:
    for clf in CLF3:
        yt,yp,yprob,pp = loocv_3d(fi, clf, weighted=True)
        print_cm(f"{fl} + {clf} | 3D weighted-1/rank", yt, yp, yprob, pp)

# ── Stage 3: Attention (hardcoded) ───────────────────────────────────────────
print("=" * 56)
print("  STAGE 3 | Gated Attention MLP | Weighted 1/rank")
print("=" * 56 + "\n")
attn = [
    ("VaRFS+Attention | 3D weighted",
     [1,1,1,0,0],[1,0,1,0,0],[0.61,0.45,0.94,0.11,0.41],
     {"Patient_1":(1,1,0.61),"Patient_2":(1,0,0.45),
      "Patient_KB":(1,1,0.94),"Patient_GA":(0,0,0.11),"Patient_VT":(0,0,0.41)}),
    ("ANOVA+Attention | 3D weighted",
     [1,1,1,0,0],[1,1,1,0,1],[0.61,0.91,0.56,0.29,0.84],
     {"Patient_1":(1,1,0.61),"Patient_2":(1,1,0.91),
      "Patient_KB":(1,1,0.56),"Patient_GA":(0,0,0.29),"Patient_VT":(0,1,0.84)}),
]
for title, yt, yp, yprob, pp in attn:
    print_cm(title, yt, yp, yprob, pp)

# ── Stage 4: 2D Slice ─────────────────────────────────────────────────────────
print("=" * 56)
print("  STAGE 4 | 2D Slice Radiomics | Weighted liver_px")
print("=" * 56 + "\n")
names2 = {"KNN":"KNN","RF":"RandomForest","GB":"GradientBoost","SVM":"SVM","MLP":"MLP"}
for clf in CLF2:
    yt,yp,yprob,pp = loocv_2d(clf)
    print_cm(f"2D-Slice + {names2[clf]} | liver_px weight", yt, yp, yprob, pp)

# ── Stage 5: 2D+3D Combined ───────────────────────────────────────────────────
print("=" * 56)
print("  STAGE 5 | 2D+3D Combined | rank-weight + thresh=0.45")
print("=" * 56 + "\n")
for fl, fi in [("VaRFS", iv23), ("ANOVA", ia23)]:
    for clf in CLF2:
        yt,yp,yprob,pp = loocv_23(fi, clf, threshold=0.45)
        print_cm(f"{fl}+{names2[clf]} | 2D+3D rank-weighted", yt, yp, yprob, pp)

# ── Stage 6: 2D CNN (hardcoded) ───────────────────────────────────────────────
print("=" * 56)
print("  STAGE 6 | 2D CNN (ResNet18, run on CPU)")
print("=" * 56 + "\n")
cnn = [
    ("SimpleCNN | 2D raw CT slices",
     [1,1,1,0,0],[1,1,1,1,1],[0.963,0.786,0.889,0.941,0.961],
     {"Patient_1":(1,1,0.963),"Patient_2":(1,1,0.786),
      "Patient_KB":(1,1,0.889),"Patient_GA":(0,1,0.941),"Patient_VT":(0,1,0.961)}),
    ("ResNet18-Linear | 2D raw CT",
     [1,1,1,0,0],[1,1,1,1,1],[0.665,0.710,0.567,0.751,0.863],
     {"Patient_1":(1,1,0.665),"Patient_2":(1,1,0.710),
      "Patient_KB":(1,1,0.567),"Patient_GA":(0,1,0.751),"Patient_VT":(0,1,0.863)}),
    ("ResNet18-SE-Attention | 2D raw CT",
     [1,1,1,0,0],[1,1,1,1,1],[0.780,0.887,0.630,0.903,0.974],
     {"Patient_1":(1,1,0.780),"Patient_2":(1,1,0.887),
      "Patient_KB":(1,1,0.630),"Patient_GA":(0,1,0.903),"Patient_VT":(0,1,0.974)}),
    ("ResNet18-FineTune | 2D raw CT",
     [1,1,1,0,0],[1,1,1,1,1],[0.742,0.815,0.869,0.976,0.928],
     {"Patient_1":(1,1,0.742),"Patient_2":(1,1,0.815),
      "Patient_KB":(1,1,0.869),"Patient_GA":(0,1,0.976),"Patient_VT":(0,1,0.928)}),
]
for title, yt, yp, yprob, pp in cnn:
    print_cm(title, yt, yp, yprob, pp)

# ── Legend ────────────────────────────────────────────────────────────────────
print("=" * 56)
print("  LEGEND")
print("=" * 56)
print("  TP = Cancer -> Cancer   (correct)")
print("  TN = Healthy -> Healthy (correct)")
print("  FP = Healthy -> Cancer  (unnecessary biopsy)")
print("  FN = Cancer -> Healthy  (missed diagnosis)")
print()
print("  Acc  = (TP+TN) / total")
print("  Prec = TP / (TP+FP)")
print("  Recall / Sens = TP / (TP+FN)")
print("  Spec = TN / (TN+FP)")
print("  F1   = 2 * Prec * Recall / (Prec + Recall)")
print("  AUC  = Area Under ROC Curve")
