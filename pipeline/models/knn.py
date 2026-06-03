"""
K-Nearest Neighbors classifier (k=3).

Why KNN won: when all 3 nearest neighbors of VT_s01 in 5D feature space
are healthy series, it returns P(Cancer)=0.000 — a hard zero.
This "polarity" is what allows the 1/rank weighting to push
the patient prediction below the 0.5 threshold.
"""
from sklearn.neighbors import KNeighborsClassifier

NAME = "KNN"


def make_clf():
    return KNeighborsClassifier(n_neighbors=3)
