"""
Random Forest classifier (100 trees, class-balanced).

Why RF failed on Patient_VT: even when a series is clearly healthy,
RF averages probability across all trees and gives ~0.41 for VT_s01.
After 1/rank weighting this is still enough to cross the 0.5 threshold.
Compare: KNN gives exactly 0.000 for the same series.
"""
from sklearn.ensemble import RandomForestClassifier

NAME = "RandomForest"


def make_clf():
    return RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
    )
