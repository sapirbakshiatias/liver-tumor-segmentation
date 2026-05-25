import sys, warnings, numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")
warnings.filterwarnings("ignore")

from train_all_series_report import (
    clean_features, cv_filter, compute_icc,
    select_varfs, select_anova, run_patient_loocv,
)

CSV_PATH = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\all_series_radiomics.csv"
df       = pd.read_csv(CSV_PATH)
feat_cols = [c for c in df.columns if c not in ("series","patient","group","label")]
groups    = df["group"].tolist()
y_all     = df["label"].values.astype(int)
X_raw     = df[feat_cols].values.astype(float)

X_clean, fnames = clean_features(X_raw, list(feat_cols))
cv_keep         = cv_filter(X_clean, fnames)
icc_vals        = compute_icc(df, fnames)

idx_varfs, f_scores, _, _ = select_varfs(X_clean, y_all, fnames, icc_vals, cv_keep)
idx_anova, _              = select_anova(X_clean, y_all, fnames, cv_keep)

clf_names    = ["RandomForest","SVM","LogisticRegr","KNN","NaiveBayes","DecisionTree"]
combinations = (
    [(f"VaRFS + {c}", idx_varfs, c) for c in clf_names] +
    [(f"ANOVA + {c}", idx_anova, c) for c in clf_names]
)

PATIENTS = ["Patient_1","Patient_2","Patient_KB","Patient_GA","Patient_VT"]
LABELS   = {"Patient_1":"C","Patient_2":"C","Patient_KB":"C","Patient_GA":"H","Patient_VT":"H"}

header = f"{'Model':<30}" + "".join(f"  {p:<13}" for p in PATIENTS)
print(header)
print("-" * len(header))

for label, feat_idx, clf_name in combinations:
    _, _, _, _, _, fold_log = run_patient_loocv(
        X_clean, y_all, groups, feat_idx, clf_name)
    row = {p: (tl, pl, pr) for p, tl, pl, pr in fold_log}
    line = f"{label:<30}"
    for p in PATIENTS:
        tl, pl, pr = row[p]
        status = "OK" if tl == pl else "WRONG"
        line += f"  {status}({pr:.2f})   "
    print(line)
