"""
Logistic Regression.

מודל לינארי פשוט — מחשב הסתברות לפי צירוף לינארי של הפיצ'רים.
max_iter=1000: יותר איטרציות לכינוס עם נתונים קטנים.
"""
from sklearn.linear_model import LogisticRegression

NAME = "LogisticRegr"

def make_clf():
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
