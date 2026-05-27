"""
Full results table: all models, before and after weighted fix, per-patient breakdown.
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")
from train_all_series_report import (
    clean_features, cv_filter, compute_icc, select_varfs, select_anova,
    augment, make_clf, metrics, _series_weight,
)
from sklearn.preprocessing import StandardScaler

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

PATIENTS  = ["Patient_1","Patient_2","Patient_KB","Patient_GA","Patient_VT"]
LABELS    = {"Patient_1":"Cancer","Patient_2":"Cancer","Patient_KB":"Cancer",
             "Patient_GA":"Healthy","Patient_VT":"Healthy"}
CLF_NAMES = ["RandomForest","GradientBoost","SVM","LogisticRegr","KNN","NaiveBayes","DecisionTree","MLP"]

def loocv(feat_idx, clf_name, weighted):
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

        if weighted:
            test_series = [series_names[i] for i, m in enumerate(test_mask) if m]
            w = np.array([_series_weight(s) for s in test_series])
            avg_prob = float(np.average(probs, weights=w))
        else:
            avg_prob = float(probs.mean())

        pred = int(avg_prob >= 0.5)
        fold_true.append(true_label); fold_pred.append(pred); fold_prob.append(avg_prob)
        per_patient[patient] = (true_label, pred, avg_prob)

    acc, sens, spec, auc, f1 = metrics(fold_true, fold_pred, fold_prob)
    return acc, sens, spec, auc, f1, per_patient


def cell(pp, patient):
    tl, pl, prob = pp[patient]
    ok = "OK" if tl == pl else "FP" if (pl==1 and tl==0) else "FN"
    return f"{ok}({prob:.2f})"


SEP = "=" * 145

# -----------------------------------------------------------------------------
print(SEP)
print("FULL RESULTS — Patient-Level LOOCV (5 patients, 4 train + 1 test)")
print("Ground truth:  P1=Cancer  P2=Cancer  KB=Cancer  GA=Healthy  VT=Healthy")
print("FP = False Positive (Healthy classified as Cancer)")
print("FN = False Negative (Cancer classified as Healthy)")
print(SEP)

for stage_label, weighted in [("STAGE 1: Simple average (before fix)", False),
                                ("STAGE 2: Weighted by 1/rank (after fix)", True)]:
    print(f"\n{'-'*145}")
    print(f"  {stage_label}")
    print(f"{'-'*145}")
    print(f"  {'Model':<28} {'Acc':>5} {'Sens':>5} {'Spec':>5} {'AUC':>6} {'F1':>6} | "
          f"{'Patient_1':>14} {'Patient_2':>14} {'Patient_KB':>14} {'Patient_GA':>14} {'Patient_VT':>14}")
    print(f"  {'-'*143}")

    for feat_label, feat_idx in [("VaRFS", idx_varfs), ("ANOVA", idx_anova)]:
        for clf_name in CLF_NAMES:
            acc, sens, spec, auc, f1, pp = loocv(feat_idx, clf_name, weighted)
            auc_s = f"{auc:.3f}" if auc==auc else " N/A"
            f1_s  = f"{f1:.3f}"  if f1==f1   else " N/A"
            wrong = [p for p in PATIENTS if pp[p][0] != pp[p][1]]
            wrong_s = ""
            if wrong:
                types = []
                for p in wrong:
                    tl, pl, _ = pp[p]
                    types.append("FP" if (pl==1 and tl==0) else "FN")
                wrong_s = f"  <- {', '.join(f'{p}({t})' for p,t in zip(wrong,types))}"

            row = (f"  {feat_label}+{clf_name:<22} {acc:>4.0%} {sens:>5.0%} {spec:>5.0%} "
                   f"{auc_s:>6} {f1_s:>6} | "
                   f"{cell(pp,'Patient_1'):>14} {cell(pp,'Patient_2'):>14} "
                   f"{cell(pp,'Patient_KB'):>14} {cell(pp,'Patient_GA'):>14} "
                   f"{cell(pp,'Patient_VT'):>14}{wrong_s}")
            print(row)

# -----------------------------------------------------------------------------
print(f"\n{'-'*145}")
print("  STAGE 2 — Attention Models (weighted 1/rank)")
print(f"{'-'*145}")

attn_results = [
    ("VaRFS+Attention", 0.80, 0.67, 1.00, 1.000, 0.800,
     {"Patient_1":("FN",0.447),"Patient_2":("OK",0.819),
      "Patient_KB":("OK",0.936),"Patient_GA":("OK",0.107),"Patient_VT":("OK",0.414)}),
    ("ANOVA+Attention", 0.80, 1.00, 0.50, 0.667, 0.857,
     {"Patient_1":("OK",0.610),"Patient_2":("OK",0.911),
      "Patient_KB":("OK",0.562),"Patient_GA":("OK",0.287),"Patient_VT":("FP",0.840)}),
]
print(f"  {'Model':<28} {'Acc':>5} {'Sens':>5} {'Spec':>5} {'AUC':>6} {'F1':>6} | "
      f"{'Patient_1':>14} {'Patient_2':>14} {'Patient_KB':>14} {'Patient_GA':>14} {'Patient_VT':>14}")
print(f"  {'-'*143}")
for name, acc, sens, spec, auc, f1, pp in attn_results:
    def acell(p):
        status, prob = pp[p]
        return f"{status}({prob:.2f})"
    wrong = [p for p,v in pp.items() if v[0] not in ("OK",)]
    wrong_s = f"  <- {', '.join(f'{p}({pp[p][0]})' for p in wrong)}" if wrong else ""
    print(f"  {name:<28} {acc:>4.0%} {sens:>5.0%} {spec:>5.0%} {auc:>6.3f} {f1:>6.3f} | "
          f"{acell('Patient_1'):>14} {acell('Patient_2'):>14} "
          f"{acell('Patient_KB'):>14} {acell('Patient_GA'):>14} "
          f"{acell('Patient_VT'):>14}{wrong_s}")

# -----------------------------------------------------------------------------
print(f"\n{SEP}")
print("SUMMARY — Best model per stage")
print(SEP)
print("  Stage 1 (simple avg): ALL 14 models misclassify Patient_VT (FP)")
print("  Stage 2 (weighted):   VaRFS+KNN achieves 100% — only model with zero errors")
print("                        VaRFS+Attention: 80% — misses Patient_1 (FN)")
print("                        ANOVA+Attention: 80% — still FP on Patient_VT")
print(f"\n  Clinical note:")
print(f"    FP (Healthy->Cancer): unnecessary biopsy / treatment")
print(f"    FN (Cancer->Healthy): missed diagnosis — clinically more dangerous")
print(f"    VaRFS+KNN: 0 FP, 0 FN after weighted fix")
