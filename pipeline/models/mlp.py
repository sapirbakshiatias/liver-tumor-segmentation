"""
Multi-Layer Perceptron — two hidden layers (32 -> 16).

early_stopping monitors a 15% validation split and halts training
when validation loss stops improving, preventing overfitting.
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
