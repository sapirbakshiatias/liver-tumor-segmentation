"""
בחירת פיצ'רים: VaRFS ו-ANOVA.

VaRFS  — בוחר לפי F-score × ICC (מבדיל + עקבי)
ANOVA  — בוחר לפי F-score בלבד (מבדיל בלבד)
"""
import numpy as np
from sklearn.feature_selection import f_classif

N_FEATURES = 5


def select_varfs(X, y, fnames, icc_vals, cv_keep, k=N_FEATURES):
    """VaRFS: top-k פיצ'רים לפי F-score מנורמל × ICC."""
    f_scores, _ = f_classif(X, y)
    f_scores     = np.nan_to_num(f_scores)
    f_norm       = f_scores / (f_scores.max() + 1e-10)
    score        = f_norm * icc_vals
    score[~cv_keep] = 0.0
    top = np.argsort(score)[::-1][:k]
    return top, f_scores, icc_vals, score


def select_anova(X, y, fnames, cv_keep, k=N_FEATURES):
    """ANOVA: top-k פיצ'רים לפי F-score בלבד."""
    f_scores, _ = f_classif(X, y)
    f_scores     = np.nan_to_num(f_scores)
    f2           = f_scores.copy()
    f2[~cv_keep] = 0.0
    top = np.argsort(f2)[::-1][:k]
    return top, f_scores
