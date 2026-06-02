"""
Multi-Layer Perceptron (רשת נוירונים קלאסית).

שתי שכבות נסתרות: 32 נוירונים ואז 16.
early_stopping: עוצר לפני overfitting לפי validation loss.
validation_fraction=0.15: 15% מה-train לבדיקת עצירה מוקדמת.
"""
from sklearn.neural_network import MLPClassifier

NAME = "MLP"

def make_clf():
    return MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
    )
