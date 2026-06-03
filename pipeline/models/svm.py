"""
Support Vector Machine with RBF kernel.

Finds the maximum-margin hyperplane separating Cancer from Healthy
in the 5-dimensional feature space.
probability=True enables Platt scaling to convert decision scores to probabilities.
"""
from sklearn.svm import SVC

NAME = "SVM"


def make_clf():
    return SVC(
        kernel="rbf",
        C=1.0,
        class_weight="balanced",
        probability=True,
        random_state=42,
    )
