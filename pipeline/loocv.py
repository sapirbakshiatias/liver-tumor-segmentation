"""
Leave-One-Patient-Out Cross Validation.

בכל fold: כל הסדרות של מטופל אחד = test, השאר = train.
ניבוי מטופל = ממוצע משוקלל של הסתברויות הסדרות (1/rank).
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

from pipeline.augment       import augment
from pipeline.metrics       import metrics
from pipeline.series_weight import series_weight
from pipeline.models        import make_clf


def run_patient_loocv(X, y, groups, feat_idx, clf_name,
                      series_names=None, weighted=True):
    unique_patients = list(dict.fromkeys(groups))
    val_true, val_pred, val_prob = [], [], []
    fold_log = []

    for patient in unique_patients:
        test_mask  = np.array([g == patient for g in groups])
        train_mask = ~test_mask

        Xf_tr = X[train_mask][:, feat_idx]
        Xf_te = X[test_mask][:, feat_idx]
        y_tr  = y[train_mask]
        true_label = int(np.round(y[test_mask].mean()))

        scaler  = StandardScaler()
        Xf_tr_s = scaler.fit_transform(Xf_tr)
        Xf_te_s = scaler.transform(Xf_te)
        Xf_aug, y_aug = augment(Xf_tr_s, y_tr)

        if len(np.unique(y_aug)) < 2:
            only = int(y_aug[0])
            val_true.append(true_label); val_pred.append(only); val_prob.append(float(only))
            fold_log.append((patient, true_label, only, float(only)))
            continue

        clf = make_clf(clf_name)
        clf.fit(Xf_aug, y_aug)
        proba = clf.predict_proba(Xf_te_s)
        probs = proba[:, 1] if proba.shape[1] == 2 else np.full(len(Xf_te_s), 0.5)

        if weighted and series_names is not None:
            test_series = [series_names[i] for i, m in enumerate(test_mask) if m]
            weights     = np.array([series_weight(s) for s in test_series])
            avg_prob    = float(np.average(probs, weights=weights))
        else:
            avg_prob = float(probs.mean())

        pred = int(avg_prob >= 0.5)
        val_true.append(true_label); val_pred.append(pred); val_prob.append(avg_prob)
        fold_log.append((patient, true_label, pred, avg_prob))

    acc, sens, spec, auc, f1 = metrics(val_true, val_pred, val_prob)
    return acc, sens, spec, auc, f1, fold_log
