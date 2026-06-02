"""
Run Gated Attention MLP on all_series_radiomics.csv
with patient-level LOOCV (4 train + 1 test).

Shows:
  - Per-patient prediction for each fold
  - Attention weights (what the model focused on)
  - Comparison with classical models
"""

import sys, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")
warnings.filterwarnings("ignore")
torch.manual_seed(42)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler

from train_all_series_report import (
    clean_features, cv_filter, compute_icc,
    select_varfs, select_anova, augment, metrics,
    _series_weight,
    OUT_DIR,
)

CSV_PATH = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\all_series_radiomics.csv"
EPOCHS   = 400
LR       = 0.005
N_RUNS   = 5   # average over multiple random seeds for stability


# ── Model ─────────────────────────────────────────────────────────────────────

class GatedAttentionMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.attn = nn.Linear(n_features, n_features)
        self.gate = nn.Linear(n_features, n_features)
        self.fc1  = nn.Linear(n_features, 32)
        self.fc2  = nn.Linear(32, 16)
        self.fc3  = nn.Linear(16, 1)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        a = torch.softmax(self.attn(x), dim=-1)   # attention weights
        g = torch.sigmoid(self.gate(x))            # gating
        h = a * g * x                              # gated weighted input
        h = torch.relu(self.fc1(h))
        h = self.drop(h)
        h = torch.relu(self.fc2(h))
        return self.fc3(h)

    def get_attention(self, x_t):
        with torch.no_grad():
            a = torch.softmax(self.attn(x_t), dim=-1)
            g = torch.sigmoid(self.gate(x_t))
            return (a * g).mean(dim=0).numpy()


def train_one(X_np, y_np, seed=42):
    torch.manual_seed(seed)
    model     = GatedAttentionMLP(X_np.shape[1])
    n_pos     = max((y_np == 1).sum(), 1)
    n_neg     = max((y_np == 0).sum(), 1)
    pos_w     = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

    return model


# ── Patient-Level LOOCV ───────────────────────────────────────────────────────

def run_patient_loocv_attention(X, y, groups, feat_idx, series_names=None, weighted=True):
    unique_patients = list(dict.fromkeys(groups))
    val_true, val_pred, val_prob = [], [], []
    fold_log   = []
    fold_attn  = []

    for patient in unique_patients:
        test_mask  = np.array([g == patient for g in groups])
        train_mask = ~test_mask

        Xf_tr = X[train_mask][:, feat_idx]
        Xf_te = X[test_mask][:, feat_idx]
        y_tr  = y[train_mask]
        true_label = int(np.round(y[test_mask].mean()))

        scaler   = StandardScaler()
        Xf_tr_s  = scaler.fit_transform(Xf_tr)
        Xf_te_s  = scaler.transform(Xf_te)
        Xf_aug, y_aug = augment(Xf_tr_s, y_tr)

        # Average over N_RUNS seeds for stable probabilities
        probs_runs = []
        attn_runs  = []
        for seed in range(N_RUNS):
            model = train_one(Xf_aug, y_aug, seed=seed)
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(Xf_te_s, dtype=torch.float32)).squeeze(1)
                probs_run = torch.sigmoid(logits).numpy()
            probs_runs.append(probs_run)
            attn_runs.append(model.get_attention(
                torch.tensor(Xf_tr_s, dtype=torch.float32)))

        # Per-series probs (one per seed averaged)
        per_series_probs = np.mean(probs_runs, axis=0)  # shape: (n_test_series,)

        if weighted and series_names is not None:
            test_series = [series_names[i] for i, m in enumerate(test_mask) if m]
            weights     = np.array([_series_weight(s) for s in test_series])
            avg_prob    = float(np.average(per_series_probs, weights=weights))
        else:
            avg_prob = float(per_series_probs.mean())

        pred      = int(avg_prob >= 0.5)
        mean_attn = np.mean(attn_runs, axis=0)

        val_true.append(true_label)
        val_pred.append(pred)
        val_prob.append(avg_prob)
        fold_log.append((patient, true_label, pred, avg_prob))
        fold_attn.append(mean_attn)

    acc, sens, spec, auc, f1 = metrics(val_true, val_pred, val_prob)
    return acc, sens, spec, auc, f1, fold_log, np.mean(fold_attn, axis=0)


# ── Attention weight bar chart ────────────────────────────────────────────────

def plot_attention_weights(feat_names_varfs, attn_varfs,
                           feat_names_anova, attn_anova, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#E53935", "#FB8C00", "#FDD835", "#43A047", "#1E88E5"]

    for ax, feat_names, attn, title in [
        (ax1, feat_names_varfs, attn_varfs, "VaRFS + Attention"),
        (ax2, feat_names_anova, attn_anova, "ANOVA + Attention"),
    ]:
        order  = np.argsort(attn)[::-1]
        names  = [feat_names[i].replace("_", " ") for i in order]
        values = [attn[i] for i in order]
        bars   = ax.barh(names[::-1], values[::-1],
                         color=colors[:len(names)][::-1], edgecolor="white")
        for bar, val in zip(bars, values[::-1]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9)
        ax.set_xlabel("Attention weight (attn × gate)", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(0, max(values) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("What the Attention model learned to focus on",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df           = pd.read_csv(CSV_PATH)
    feat_cols    = [c for c in df.columns if c not in ("series","patient","group","label")]
    groups       = df["group"].tolist()
    series_names = df["series"].tolist()
    y_all        = df["label"].values.astype(int)
    X_raw        = df[feat_cols].values.astype(float)

    X_clean, fnames = clean_features(X_raw, list(feat_cols))
    cv_keep         = cv_filter(X_clean, fnames)
    icc_vals        = compute_icc(df, fnames)

    idx_varfs, f_scores, _, _ = select_varfs(X_clean, y_all, fnames, icc_vals, cv_keep)
    idx_anova, _              = select_anova(X_clean, y_all, fnames, cv_keep)

    varfs_feats = [fnames[i] for i in idx_varfs]
    anova_feats = [fnames[i] for i in idx_anova]

    print("=" * 60)
    print("GATED ATTENTION MLP — Patient-Level LOOCV")
    print(f"Architecture: GatedAttn -> Linear(32) -> Linear(16) -> 1")
    print(f"Epochs: {EPOCHS}  LR: {LR}  Averaged over {N_RUNS} seeds")
    print("=" * 60)
    print(f"\nVaRFS features: {varfs_feats}")
    print(f"ANOVA features: {anova_feats}")

    results = []
    for label, feat_idx, feat_names in [
        ("VaRFS + Attention", idx_varfs, varfs_feats),
        ("ANOVA + Attention", idx_anova, anova_feats),
    ]:
        print(f"\n{'='*60}")
        print(f"Running: {label}")
        print(f"{'='*60}")

        acc, sens, spec, auc, f1, fold_log, mean_attn = \
            run_patient_loocv_attention(X_clean, y_all, groups, feat_idx,
                                        series_names=series_names, weighted=True)

        print(f"\nPer-patient results:")
        print(f"  {'Patient':<15} {'True':>8} {'Pred':>8} {'Prob':>7} {'Result'}")
        print(f"  {'-'*55}")
        for patient, true_l, pred_l, prob in fold_log:
            status = "OK   " if true_l == pred_l else "WRONG"
            true_s = "Cancer"  if true_l == 1 else "Healthy"
            pred_s = "Cancer"  if pred_l == 1 else "Healthy"
            print(f"  {patient:<15} {true_s:>8} {pred_s:>8} {prob:>7.3f}  {status}")

        a_s = f"{auc:.3f}" if not (auc != auc) else "N/A"
        f_s = f"{f1:.3f}"  if not (f1 != f1)   else "N/A"
        print(f"\n  => Acc={acc:.0%}  Sens={sens:.0%}  Spec={spec:.0%}  "
              f"AUC={a_s}  F1={f_s}")

        print(f"\nAttention weights (higher = model focuses more on this feature):")
        order = np.argsort(mean_attn)[::-1]
        for rank, i in enumerate(order):
            bar = "#" * int(mean_attn[i] * 80)
            print(f"  {rank+1}. {feat_names[i]:<40}  {mean_attn[i]:.4f}  {bar}")

        results.append({
            "Model": label, "Acc": acc, "Sens": sens, "Spec": spec,
            "AUC": auc, "F1": f1, "fold_log": fold_log,
            "attn": mean_attn, "feats": feat_names,
        })

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'Acc':>5} {'Sens':>6} {'Spec':>6} {'AUC':>7} {'F1':>7}")
    print(f"  {'-'*55}")
    for r in results:
        a_s = f"{r['AUC']:.3f}" if not (r['AUC'] != r['AUC']) else "  N/A"
        f_s = f"{r['F1']:.3f}"  if not (r['F1']  != r['F1'])  else "  N/A"
        print(f"  {r['Model']:<25} {r['Acc']:>4.0%}  {r['Sens']:>5.0%}  "
              f"{r['Spec']:>5.0%}  {a_s:>6}  {f_s:>6}")

    # Attention weight chart
    print("\nGenerating attention weight chart...")
    import os
    plot_attention_weights(
        results[0]["feats"], results[0]["attn"],
        results[1]["feats"], results[1]["attn"],
        os.path.join(OUT_DIR, "attention_weights.png"),
    )

    print(f"\nDone. Chart saved to: {OUT_DIR}")
