"""
Support Vector Machine — RBF kernel.

מחפש hyperplane שמפריד בין קבוצות במרחב הפיצ'רים.
C=1.0: איזון בין margin רחב לשגיאות אימון.
probability=True: ממיר decision function להסתברויות (Platt scaling).
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
