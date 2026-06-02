"""
Decision Tree.

עץ החלטה בעומק מקסימלי 3 — ליצור כללי IF-THEN פשוטים.
max_depth=3: מגביל overfitting עם מעט נתונים.
"""
from sklearn.tree import DecisionTreeClassifier

NAME = "DecisionTree"

def make_clf():
    return DecisionTreeClassifier(
        max_depth=3,
        class_weight="balanced",
        random_state=42,
    )
