"""
Seaborn heatmap סטטי: שורות = סדרות, עמודות = פיצ'רים, ערכים Z-normalized.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

COLOR_CANCER  = "#C62828"
COLOR_HEALTHY = "#1B5E20"


def plot_heatmap(df, feat_names, title, out_path):
    sub = df[["series", "group", "label"] + feat_names].copy()
    sub = sub.sort_values(["label", "group"], ascending=[False, True])

    X      = sub[feat_names].values.astype(float)
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    row_labels = [f"{r.series}  ({'C' if r.label==1 else 'H'})" for r in sub.itertuples()]
    row_colors = [COLOR_CANCER if r.label == 1 else COLOR_HEALTHY for r in sub.itertuples()]

    fig, ax = plt.subplots(figsize=(max(10, len(feat_names)*2), max(6, len(sub)*0.45)))
    sns.heatmap(
        pd.DataFrame(X_norm, columns=feat_names, index=row_labels),
        cmap="RdBu_r", center=0, linewidths=0.4, linecolor="#333333",
        ax=ax, cbar_kws={"label": "Z-score", "shrink": 0.6},
        annot=True, fmt=".2f", annot_kws={"size": 6},
    )
    for i, rc in enumerate(row_colors):
        ax.add_patch(plt.Rectangle((-0.35, i), 0.28, 1, color=rc, clip_on=False, zorder=3))

    ax.legend(handles=[
        mpatches.Patch(color=COLOR_CANCER,  label="Cancer"),
        mpatches.Patch(color=COLOR_HEALTHY, label="Healthy"),
    ], loc="upper right", fontsize=8, bbox_to_anchor=(1.18, 1.02))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Feature"); ax.set_ylabel("Series")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
