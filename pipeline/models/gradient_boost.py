"""
Gradient Boosting.

בונה עצים ברצף — כל עץ מתקן את שגיאות הקודם.
learning_rate=0.1: צעדים קטנים למניעת overfitting.
max_depth=3: עצים רדודים ליציבות.
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
