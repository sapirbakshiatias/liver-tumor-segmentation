"""
Feature selection: VaRFS and ANOVA.

VaRFS  — selects by normalized F-score x ICC (discriminative AND reproducible)
ANOVA  — selects by F-score only (most discriminative, ignores reproducibility)

VaRFS is the method used by the winning model (VaRFS + KNN).
The key difference: VaRFS includes sagittal_glcm_dissimilarity (ICC=0.573),
which ANOVA misses because its raw F-score is too low.
"""
import numpy as np
from sklearn.feature_selection import f_classif

N_FEATURES = 5  # number of features to select


def select_varfs(X, y, fnames, icc_vals, cv_keep, k=N_FEATURES):
    """
    VaRFS: select top-k features by normalized_F_score x ICC.

    A feature with moderate F-score but high ICC can outrank
    a high-F-score feature that is unstable across scans.
    """
    f_scores, _ = f_classif(X, y)
    f_scores     = np.nan_to_num(f_scores)
    f_norm       = f_scores / (f_scores.max() + 1e-10)
    score        = f_norm * icc_vals
    score[~cv_keep] = 0.0  # exclude high-CV (unstable) features
    top = np.argsort(score)[::-1][:k]
    return top, f_scores, icc_vals, score


def select_anova(X, y, fnames, cv_keep, k=N_FEATURES):
    """
    ANOVA: select top-k features by F-score only.
    Simpler than VaRFS — does not account for reproducibility.
    """
    f_scores, _ = f_classif(X, y)
    f_scores     = np.nan_to_num(f_scores)
    f2           = f_scores.copy()
    f2[~cv_keep] = 0.0
    top = np.argsort(f2)[::-1][:k]
    return top, f_scores
