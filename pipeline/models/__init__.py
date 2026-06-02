from .knn               import make_clf as make_knn
from .random_forest     import make_clf as make_rf
from .svm               import make_clf as make_svm
from .logistic_regression import make_clf as make_lr
from .naive_bayes       import make_clf as make_nb
from .decision_tree     import make_clf as make_dt
from .gradient_boost    import make_clf as make_gb
from .mlp               import make_clf as make_mlp

ALL_MODELS = {
    "KNN":           make_knn,
    "RandomForest":  make_rf,
    "SVM":           make_svm,
    "LogisticRegr":  make_lr,
    "NaiveBayes":    make_nb,
    "DecisionTree":  make_dt,
    "GradientBoost": make_gb,
    "MLP":           make_mlp,
}

def make_clf(name):
    if name not in ALL_MODELS:
        raise ValueError(f"Unknown model: {name}. Options: {list(ALL_MODELS)}")
    return ALL_MODELS[name]()
