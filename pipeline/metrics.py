"""
מדדי ביצוע: Accuracy, Sensitivity, Specificity, AUC, F1.

TP = Cancer מסווג נכון    | FP = Healthy → Cancer  (ביופסיה מיותרת)
TN = Healthy מסווג נכון   | FN = Cancer  → Healthy (פספוס מסוכן)
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def metrics(y_true, y_pred, y_prob=None):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc  = (tp + tn) / len(y_true)
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1   = (2*prec*sens / (prec+sens)
            if not any(map(lambda v: v != v, [prec, sens])) and prec+sens > 0
            else float("nan"))
    auc = float("nan")
    if y_prob is not None and len(set(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            pass
    return acc, sens, spec, auc, f1
