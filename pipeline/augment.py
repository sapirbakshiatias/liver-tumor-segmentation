"""
Class imbalance handling via oversampling.

SMOTE creates synthetic minority samples along the line between real neighbors.
RandomOverSampler duplicates existing samples when only one minority sample exists.
"""
from imblearn.over_sampling import SMOTE, RandomOverSampler


def augment(X, y):
    """Oversample the minority class (healthy patients) to balance training data."""
    n_min = (y == 0).sum()
    if n_min >= 2:
        return SMOTE(k_neighbors=1, random_state=42).fit_resample(X, y)
    if n_min == 1:
        return RandomOverSampler(random_state=42).fit_resample(X, y)
    return X, y
