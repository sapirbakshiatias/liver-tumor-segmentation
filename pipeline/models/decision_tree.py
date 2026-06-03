"""
Decision Tree — interpretable IF-THEN rules.

max_depth=3 limits overfitting with small data (n=28 series).
The tree can be inspected to see which feature splits discriminate best.
"""
from sklearn.tree import DecisionTreeClassifier

NAME = "DecisionTree"


def make_clf():
    return DecisionTreeClassifier(
        max_depth=3,
        class_weight="balanced",
        random_state=42,
    )
