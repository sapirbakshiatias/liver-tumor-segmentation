"""
K-Nearest Neighbors classifier.

k=3: לכל נקודת test מחפש 3 שכנים קרובים ביותר בחלל הפיצ'רים.
הסיבה ש-KNN מנצח: כשכל 3 שכנים בריאים הוא נותן 0.000 — קוטבי לחלוטין.
"""
from sklearn.neighbors import KNeighborsClassifier

NAME = "KNN"

def make_clf():
    return KNeighborsClassifier(n_neighbors=3)
