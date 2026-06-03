"""
Gradient Boosting — sequential ensemble of shallow trees.

Each tree corrects the residual errors of the previous one.
learning_rate=0.1 and max_depth=3 prevent overfitting on small data.
"""
from sklearn.ensemble import GradientBoostingClassifier

NAME = "GradientBoost"


def make_clf():
    return GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
