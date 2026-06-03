"""
Logistic Regression — linear classifier.

Computes cancer probability as a sigmoid of a linear combination of features.
With rank-based weighting, VaRFS+LogisticRegr starts misclassifying
Patient_1 and Patient_KB as FN — the linear boundary is less robust
than KNN's local neighborhood decision.
"""
from sklearn.linear_model import LogisticRegression

NAME = "LogisticRegr"


def make_clf():
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
