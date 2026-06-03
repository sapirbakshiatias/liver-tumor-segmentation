"""
Gaussian Naive Bayes.

Assumes each feature follows a Gaussian distribution within each class,
and that features are conditionally independent given the class.
No hyperparameters. AUC=0.5 in most folds — the independence assumption
is violated here (texture features are correlated).
"""
from sklearn.naive_bayes import GaussianNB

NAME = "NaiveBayes"


def make_clf():
    return GaussianNB()
