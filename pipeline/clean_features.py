"""
ניקוי פיצ'רים: הסרת NaN, קבועים, וחיתוך outliers.
"""
import numpy as np

CV_THRESHOLD = 100.0


def clean_features(X, feature_names):
    """מסיר NaN/Inf, פיצ'רים קבועים, וחותך outliers ב-±3 סטיות תקן."""
    bad = np.any(~np.isfinite(X), axis=0)
    X   = X[:, ~bad]
    names = [f for f, b in zip(feature_names, bad) if not b]

    std   = X.std(axis=0)
    const = std == 0
    X     = X[:, ~const]
    names = [f for f, c in zip(names, const) if not c]

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    X  = np.clip(X, mu - 3*sd, mu + 3*sd)
    return X, names


def cv_filter(X, feature_names):
    """מחזיר מסיכה: True = פיצ'ר עם CV סביר (לא רועש מדי)."""
    mean = np.abs(X.mean(axis=0))
    std  = X.std(axis=0)
    cv   = np.where(mean > 1e-10, std / mean * 100, np.inf)
    return cv <= CV_THRESHOLD
