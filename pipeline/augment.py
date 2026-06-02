"""
Augmentation — איזון אי-שיווי מעמד בין קבוצות.
SMOTE יוצר דוגמאות סינתטיות, RandomOverSampler כופל.
"""
from imblearn.over_sampling import SMOTE, RandomOverSampler


def augment(X, y):
    n_min = (y == 0).sum()
    if n_min >= 2:
        return SMOTE(k_neighbors=1, random_state=42).fit_resample(X, y)
    if n_min == 1:
        return RandomOverSampler(random_state=42).fit_resample(X, y)
    return X, y
