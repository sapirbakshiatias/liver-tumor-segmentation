import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")
from train_all_series_report import (
    clean_features, cv_filter, compute_icc, select_varfs, augment, make_clf, _series_weight
)
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\all_series_radiomics.csv")
feat_cols    = [c for c in df.columns if c not in ("series","patient","group","label")]
groups       = df["group"].tolist()
series_names = df["series"].tolist()
y_all        = df["label"].values.astype(int)
X_raw        = df[feat_cols].values.astype(float)

X_clean, fnames = clean_features(X_raw, list(feat_cols))
cv_keep         = cv_filter(X_clean, fnames)
icc_vals        = compute_icc(df, fnames)
idx_varfs, _, _, _ = select_varfs(X_clean, y_all, fnames, icc_vals, cv_keep)

test_mask  = np.array([g == "Patient_VT" for g in groups])
train_mask = ~test_mask

Xf_tr = X_clean[train_mask][:, idx_varfs]
Xf_te = X_clean[test_mask][:, idx_varfs]
y_tr  = y_all[train_mask]

scaler  = StandardScaler()
Xf_tr_s = scaler.fit_transform(Xf_tr)
Xf_te_s = scaler.transform(Xf_te)
Xf_aug, y_aug = augment(Xf_tr_s, y_tr)

test_series = [series_names[i] for i, m in enumerate(test_mask) if m]
weights     = [_series_weight(s) for s in test_series]

rf  = make_clf("RandomForest"); rf.fit(Xf_aug, y_aug)
knn = make_clf("KNN");          knn.fit(Xf_aug, y_aug)

rf_probs  = rf.predict_proba(Xf_te_s)[:, 1]
knn_probs = knn.predict_proba(Xf_te_s)[:, 1]

print("Per-series probabilities for Patient_VT (VaRFS features):")
print(f"  {'Series':<35} {'Weight':>8}    RF   KNN")
print(f"  {'-'*60}")
for s, w, rp, kp in zip(test_series, weights, rf_probs, knn_probs):
    rf_flag  = " <-- pulls up" if rp >= 0.5 else ""
    knn_flag = " <-- pulls up" if kp >= 0.5 else ""
    print(f"  {s:<35} {w:>8.3f}  {rp:.3f}{rf_flag}   {kp:.3f}{knn_flag}")

rf_wavg  = float(np.average(rf_probs,  weights=weights))
knn_wavg = float(np.average(knn_probs, weights=weights))
print(f"\n  Weighted avg:                                     RF={rf_wavg:.3f}  KNN={knn_wavg:.3f}")
print(f"  Decision:                                          {'Cancer' if rf_wavg>=0.5 else 'Healthy'}       {'Cancer' if knn_wavg>=0.5 else 'Healthy'}")

# Also show the s01 feature values vs training cancer/healthy means
varfs_feats = [fnames[i] for i in idx_varfs]
cancer_mask  = y_all == 1
healthy_mask = y_all == 0

print("\nVT s01 feature values vs group means:")
print(f"  {'Feature':<35} {'Cancer':>10} {'Healthy':>10} {'VT s01':>10}  Closer to:")
s01_idx = next(i for i, s in enumerate(test_series) if s.endswith("_s01"))
vt_s01_row = X_clean[test_mask][s01_idx]
for j, feat in enumerate(varfs_feats):
    cm = X_clean[cancer_mask, idx_varfs[j]].mean()
    hm = X_clean[healthy_mask, idx_varfs[j]].mean()
    vt = vt_s01_row[idx_varfs[j]]
    closer = "Cancer" if abs(vt - cm) < abs(vt - hm) else "Healthy"
    print(f"  {feat:<35} {cm:>10.3f} {hm:>10.3f} {vt:>10.3f}  {closer}")
