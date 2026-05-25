"""
Deep Learning approach: Gated Attention MLP for liver cancer prediction.

Architecture per fold:
  Input: k=5 radiomics features (StandardScaler normalized)
    -> Attention head: Linear(k->k) + Softmax   (how much each feature matters)
    -> Gate head:      Linear(k->k) + Sigmoid   (on/off switch per feature)
    -> h = attention * gate * input              (gated weighted features)
    -> Linear(k->16) -> ReLU -> Dropout(0.3)
    -> Linear(16->1) -> Sigmoid
    -> cancer probability

Same preprocessing as train_svm.py (clean -> CV filter -> ICC -> VaRFS/ANOVA).
Same LOOCV + SMOTE framework for fair comparison.
"""

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler

warnings.filterwarnings("ignore")
torch.manual_seed(42)

CSV_PATH = r"c:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\liver_radiomics.csv"

ICC_PAIR_NAMES = [
    ("Patient_1_Before_CT",  "Patient_1_After_CT"),
    ("Patient_2_Before_CT",  "Patient_2_After_CT"),
    ("Patient_KB_Before_CT", "Patient_KB_Cancer_CT"),
]

CV_THRESHOLD = 100.0
N_FEATURES   = 5
EPOCHS       = 300
LR           = 0.01


# ── Preprocessing (same logic as train_svm.py) ────────────────────────────────

def clean_features(X_all, feature_names, train_idx):
    X = X_all.copy().astype(float)
    bad = np.any(~np.isfinite(X), axis=0)
    X = X[:, ~bad]
    feature_names = [f for f, b in zip(feature_names, bad) if not b]
    train_std = X[train_idx].std(axis=0)
    const = train_std == 0
    X = X[:, ~const]
    feature_names = [f for f, c in zip(feature_names, const) if not c]
    mu = X[train_idx].mean(axis=0)
    sd = X[train_idx].std(axis=0)
    X  = np.clip(X, mu - 3 * sd, mu + 3 * sd)
    return X, feature_names


def cv_filter(X_train, feature_names):
    mean = np.abs(X_train.mean(axis=0))
    std  = X_train.std(axis=0)
    cv   = np.where(mean > 1e-10, std / mean * 100, np.inf)
    return cv <= CV_THRESHOLD


def compute_icc(df, feature_names):
    before_rows, after_rows = [], []
    for b_name, a_name in ICC_PAIR_NAMES:
        b = df[df["patient"] == b_name]
        a = df[df["patient"] == a_name]
        if b.empty or a.empty:
            continue
        before_rows.append(b[feature_names].values[0])
        after_rows.append(a[feature_names].values[0])
    if len(before_rows) < 2:
        return np.ones(len(feature_names))
    B, A = np.array(before_rows), np.array(after_rows)
    n, k = B.shape[0], 2
    vals        = np.stack([B, A], axis=1)
    subj_means  = vals.mean(axis=1)
    grand_means = vals.mean(axis=(0, 1))
    MSB   = k * np.sum((subj_means - grand_means) ** 2, axis=0) / (n - 1)
    MSW   = np.sum((vals - subj_means[:, None, :]) ** 2, axis=(0, 1)) / (n * (k - 1))
    denom = MSB + (k - 1) * MSW
    icc   = np.where(denom > 0, (MSB - MSW) / denom, 0.0)
    return np.clip(icc, 0.0, 1.0)


def select_varfs(X_train, y_train, icc_vals, cv_keep, k=N_FEATURES):
    f_scores, _ = f_classif(X_train, y_train)
    f_scores = np.nan_to_num(f_scores)
    f_norm   = f_scores / (f_scores.max() + 1e-10)
    score    = f_norm * icc_vals
    score[~cv_keep] = 0.0
    return np.argsort(score)[::-1][:k]


def select_anova(X_train, y_train, cv_keep, k=N_FEATURES):
    f_scores, _ = f_classif(X_train, y_train)
    f_scores = np.nan_to_num(f_scores)
    f_scores[~cv_keep] = 0.0
    return np.argsort(f_scores)[::-1][:k]


def augment(X, y):
    n_min = (y == 0).sum()
    if n_min >= 2:
        return SMOTE(k_neighbors=1, random_state=42).fit_resample(X, y)
    if n_min == 1:
        return RandomOverSampler(random_state=42).fit_resample(X, y)
    return X, y


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
    f1   = (2 * prec * sens / (prec + sens)
            if not (np.isnan(prec) or np.isnan(sens)) and (prec + sens) > 0
            else float("nan"))
    auc  = float("nan")
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            pass
    return acc, sens, spec, auc, prec, f1


# ── Model ─────────────────────────────────────────────────────────────────────

class GatedAttentionMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.attn_layer = nn.Linear(n_features, n_features)
        self.gate_layer = nn.Linear(n_features, n_features)
        self.fc1        = nn.Linear(n_features, 16)
        self.fc2        = nn.Linear(16, 1)
        self.dropout    = nn.Dropout(0.3)

    def forward(self, x):
        attn = torch.softmax(self.attn_layer(x), dim=-1)   # which features matter
        gate = torch.sigmoid(self.gate_layer(x))            # on/off per feature
        h    = attn * gate * x                              # gated weighted input
        h    = torch.relu(self.fc1(h))
        h    = self.dropout(h)
        return self.fc2(h)                                  # raw logit

    def get_attention_weights(self, x_tensor):
        with torch.no_grad():
            attn = torch.softmax(self.attn_layer(x_tensor), dim=-1)
            gate = torch.sigmoid(self.gate_layer(x_tensor))
            return (attn * gate).mean(dim=0).numpy()


def train_model(X_np, y_np, verbose=False):
    n_features = X_np.shape[1]
    model      = GatedAttentionMLP(n_features)
    n_pos      = (y_np == 1).sum()
    n_neg      = (y_np == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.Adam(model.parameters(), lr=LR)

    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

    if verbose:
        print(f"\n    {'Epoch':>6}  {'Loss':>8}  {'Train Acc':>10}")
        print(f"    {'-'*30}")

    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        logits = model(X_t)
        loss   = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

        if verbose and (epoch == 1 or epoch % 50 == 0):
            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= 0.5).float()
                acc   = (preds == y_t).float().mean().item()
            print(f"    {epoch:>6}  {loss.item():>8.4f}  {acc:>9.0%}")

    return model


# ── LOOCV ─────────────────────────────────────────────────────────────────────

def run_loocv_attn(X_tr, y_tr, feat_idx):
    loo = LeaveOneOut()
    val_true, val_pred, val_prob = [], [], []
    all_attn = []

    for train_idx, test_idx in loo.split(X_tr):
        Xf_tr  = X_tr[train_idx][:, feat_idx]
        Xf_te  = X_tr[test_idx][:, feat_idx]
        y_fold = y_tr[train_idx]

        scaler   = StandardScaler()
        Xf_tr_s  = scaler.fit_transform(Xf_tr)
        Xf_te_s  = scaler.transform(Xf_te)

        Xf_aug, y_aug = augment(Xf_tr_s, y_fold)
        model = train_model(Xf_aug, y_aug)

        model.eval()
        with torch.no_grad():
            logit = model(torch.tensor(Xf_te_s, dtype=torch.float32)).item()
            prob  = float(torch.sigmoid(torch.tensor(logit)))
            pred  = int(prob >= 0.5)

        val_true.append(int(y_tr[test_idx][0]))
        val_pred.append(pred)
        val_prob.append(prob)
        all_attn.append(model.get_attention_weights(
            torch.tensor(Xf_tr_s, dtype=torch.float32)))

    acc, sens, spec, auc, prec, f1 = metrics(val_true, val_pred, val_prob)
    mean_attn = np.mean(all_attn, axis=0)
    return val_true, val_pred, val_prob, acc, sens, spec, auc, f1, mean_attn


def run_patient_loocv_attn(X_all, y_all, groups, feat_idx):
    """Leave-one-PATIENT-out for the Attention model."""
    unique_patients = list(dict.fromkeys(groups))
    val_true, val_pred, val_prob = [], [], []
    all_attn = []

    for patient in unique_patients:
        test_mask  = np.array([g == patient for g in groups])
        train_mask = ~test_mask

        Xf_tr  = X_all[train_mask][:, feat_idx]
        Xf_te  = X_all[test_mask][:, feat_idx]
        y_fold = y_all[train_mask]
        y_test = y_all[test_mask]

        scaler   = StandardScaler()
        Xf_tr_s  = scaler.fit_transform(Xf_tr)
        Xf_te_s  = scaler.transform(Xf_te)

        Xf_aug, y_aug = augment(Xf_tr_s, y_fold)
        model = train_model(Xf_aug, y_aug)

        model.eval()
        with torch.no_grad():
            logits     = model(torch.tensor(Xf_te_s, dtype=torch.float32)).squeeze(1)
            probs_fold = torch.sigmoid(logits).numpy()
            preds_fold = (probs_fold >= 0.5).astype(int)

        val_true.extend(y_test.tolist())
        val_pred.extend(preds_fold.tolist())
        val_prob.extend(probs_fold.tolist())
        all_attn.append(model.get_attention_weights(
            torch.tensor(Xf_tr_s, dtype=torch.float32)))

    acc, sens, spec, auc, prec, f1 = metrics(val_true, val_pred, val_prob)
    mean_attn = np.mean(all_attn, axis=0)
    return val_true, val_pred, val_prob, acc, sens, spec, auc, f1, mean_attn


# ── Run one combination ───────────────────────────────────────────────────────

def run_combination(label, feat_idx, fnames, X_tr, y_tr, X_te, y_te, df_train):
    print(f"\n  [{label}]")

    vt, vp, vprob, val_acc, val_sens, val_spec, val_auc, val_f1, mean_attn = \
        run_loocv_attn(X_tr, y_tr, feat_idx)

    for p, yt, yp in zip(df_train["patient"], vt, vp):
        ok = "OK   " if yt == yp else "WRONG"
        print(f"    {ok}  {p:35s}  true={'CANCER' if yt else 'HEALTHY':7s}  "
              f"pred={'CANCER' if yp else 'HEALTHY'}")

    auc_str = f"{val_auc:.3f}" if not np.isnan(val_auc) else "N/A"
    f1_str  = f"{val_f1:.3f}"  if not np.isnan(val_f1)  else "N/A"
    print(f"    LOOCV -> Acc={val_acc:.0%}  Recall={val_sens:.0%}  "
          f"Spec={val_spec:.0%}  AUC={auc_str}  F1={f1_str}")

    # Final model on all train -> test (with training curve)
    print(f"\n    --- Training curve (final model, all {len(y_tr)} train patients + SMOTE) ---")
    scaler_f = StandardScaler()
    Xf_tr_s  = scaler_f.fit_transform(X_tr[:, feat_idx])
    Xf_te_s  = scaler_f.transform(X_te[:, feat_idx])
    Xf_aug, y_aug = augment(Xf_tr_s, y_tr)
    model_f  = train_model(Xf_aug, y_aug, verbose=True)

    model_f.eval()
    with torch.no_grad():
        logits = model_f(torch.tensor(Xf_te_s, dtype=torch.float32)).squeeze(1)
        probs  = torch.sigmoid(logits).numpy()
        te_pred = (probs >= 0.5).astype(int)

    _, te_sens, te_spec, _, te_prec, te_f1 = metrics(y_te, te_pred, probs)
    print(f"    Test  -> Recall={te_sens:.0%}  F1={te_f1:.3f}  (test has no healthy patients)")

    return {
        "Combination": label,
        "LOOCV acc":  val_acc,
        "LOOCV sens": val_sens,
        "LOOCV spec": val_spec,
        "LOOCV AUC":  val_auc,
        "LOOCV F1":   val_f1,
        "Test sens":  te_sens,
        "Test F1":    te_f1,
        "Attention":  mean_attn,
        "feat_idx":   feat_idx,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    feat_cols = [c for c in df.columns if c not in ("patient", "label", "split")]

    df_train = df[df["split"] == "TRAIN"].reset_index(drop=True)
    df_test  = df[df["split"] == "TEST"].reset_index(drop=True)

    train_idx = df[df["split"] == "TRAIN"].index.tolist()
    test_idx  = df[df["split"] == "TEST"].index.tolist()

    X_all = df[feat_cols].values.astype(float)
    y_tr  = df_train["label"].values.astype(int)
    y_te  = df_test["label"].values.astype(int)

    X_clean, fnames = clean_features(X_all, list(feat_cols), train_idx)
    X_tr = X_clean[train_idx]
    X_te = X_clean[test_idx]

    cv_keep  = cv_filter(X_tr, fnames)
    icc_vals = compute_icc(df, fnames)

    idx_varfs = select_varfs(X_tr, y_tr, icc_vals, cv_keep)
    idx_anova = select_anova(X_tr, y_tr, cv_keep)

    print(f"VaRFS features: {[fnames[i] for i in idx_varfs]}")
    print(f"ANOVA features: {[fnames[i] for i in idx_anova]}")

    print("\n" + "=" * 65)
    print("GATED ATTENTION MLP — VaRFS vs ANOVA")
    print("=" * 65)

    combinations = [
        ("VaRFS + Attention", idx_varfs),
        ("ANOVA + Attention", idx_anova),
    ]

    results = []
    for label, feat_idx in combinations:
        r = run_combination(label, feat_idx, fnames,
                            X_tr, y_tr, X_te, y_te, df_train)
        results.append(r)

    # ── Summary table
    print("\n" + "=" * 65)
    print("SUMMARY TABLE")
    print("=" * 65)
    print(f"  {'Combination':<25} {'LOO-Acc':>8} {'LOO-Recall':>11} "
          f"{'LOO-Spec':>9} {'LOO-AUC':>8} {'LOO-F1':>7} {'Test-Recall':>12}")
    print("  " + "-" * 84)
    for r in results:
        auc_s = f"{r['LOOCV AUC']:.3f}" if not np.isnan(r["LOOCV AUC"]) else " N/A"
        f1_s  = f"{r['LOOCV F1']:.3f}"  if not np.isnan(r["LOOCV F1"])  else " N/A"
        print(f"  {r['Combination']:<25} {r['LOOCV acc']:>7.0%}  "
              f"{r['LOOCV sens']:>10.0%}  {r['LOOCV spec']:>8.0%}  "
              f"{auc_s:>7}  {f1_s:>6}  {r['Test sens']:>11.0%}")

    # ── Attention weights — which features the model focused on
    print("\n" + "=" * 65)
    print("TOP FEATURES BY ATTENTION WEIGHT")
    print("(what the network learned to focus on)")
    print("=" * 65)
    for r in results:
        sel   = r["feat_idx"]
        attn  = r["Attention"]
        order = np.argsort(attn)[::-1]
        print(f"\n  {r['Combination']}:")
        for rank, i in enumerate(order):
            fname = fnames[sel[i]]
            w     = attn[i]
            bar   = "#" * int(w * 50)
            print(f"    {rank+1}. {fname:<42s}  {w:.3f}  {bar}")
