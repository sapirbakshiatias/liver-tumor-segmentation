"""
כל פונקציות הגרפים והויזואליזציות.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go

from pipeline.data_preparation import OUT_DIR, sliding_feature_map, load_best_axial_slice

COLOR_CANCER  = "#C62828"
COLOR_HEALTHY = "#1B5E20"


def plot_feature_importance_bar(fnames, f_scores, icc_vals, varfs_idx, anova_idx, out_path):
    """גרף עמודות: F-score ו-ICC לכל פיצ'ר, עם צבע לפי שיטת בחירה."""
    varfs_set = set(varfs_idx)
    anova_set = set(anova_idx)
    colors = []
    for i in range(len(fnames)):
        if i in varfs_set and i in anova_set:
            colors.append("#FF6F00")
        elif i in varfs_set:
            colors.append("#1565C0")
        elif i in anova_set:
            colors.append("#2E7D32")
        else:
            colors.append("#9E9E9E")

    sort_idx       = np.argsort(f_scores)[::-1]
    sorted_names   = [fnames[i] for i in sort_idx]
    sorted_fscores = f_scores[sort_idx]
    sorted_icc     = icc_vals[sort_idx]
    sorted_colors  = [colors[i] for i in sort_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.barh(sorted_names, sorted_fscores, color=sorted_colors)
    ax1.set_xlabel("F-score (ANOVA)", fontsize=10)
    ax1.set_title("Feature F-scores", fontsize=12, fontweight="bold")
    ax1.invert_yaxis()

    ax2.barh(sorted_names, sorted_icc, color=sorted_colors)
    ax2.set_xlabel("ICC (reproducibility)", fontsize=10)
    ax2.set_title("Feature ICC", fontsize=12, fontweight="bold")
    ax2.invert_yaxis()

    legend_patches = [
        mpatches.Patch(color="#FF6F00", label="Selected by both"),
        mpatches.Patch(color="#1565C0", label="VaRFS only"),
        mpatches.Patch(color="#2E7D32", label="ANOVA only"),
        mpatches.Patch(color="#9E9E9E", label="Not selected"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Feature Importance: F-score vs ICC", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_statistical_heatmap(df, feat_names, title, out_path):
    """Seaborn heatmap: שורות = סדרות, עמודות = פיצ'רים, Z-normalized."""
    sub = df[["series", "group", "label"] + feat_names].copy()
    sub = sub.sort_values(["label", "group"], ascending=[False, True])

    X      = sub[feat_names].values.astype(float)
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    row_labels = [f"{r.series}  ({'C' if r.label==1 else 'H'})"
                  for r in sub.itertuples()]
    row_colors = [COLOR_CANCER if r.label == 1 else COLOR_HEALTHY
                  for r in sub.itertuples()]

    fig, ax = plt.subplots(figsize=(max(10, len(feat_names)*2),
                                    max(6, len(sub)*0.45)))
    sns.heatmap(
        pd.DataFrame(X_norm, columns=feat_names, index=row_labels),
        cmap="RdBu_r", center=0, linewidths=0.4, linecolor="#333333",
        ax=ax, cbar_kws={"label": "Z-score", "shrink": 0.6},
        annot=True, fmt=".2f", annot_kws={"size": 6},
    )
    for i, rc in enumerate(row_colors):
        ax.add_patch(plt.Rectangle((-0.35, i), 0.28, 1, color=rc, clip_on=False, zorder=3))

    patches = [
        mpatches.Patch(color=COLOR_CANCER,  label="Cancer"),
        mpatches.Patch(color=COLOR_HEALTHY, label="Healthy"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8, bbox_to_anchor=(1.18, 1.02))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Feature", fontsize=10)
    ax.set_ylabel("Series", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_interactive_heatmap(df, feat_names, out_path):
    """Plotly heatmap אינטרקטיבי עם hover info."""
    sub = df[["series", "group", "label"] + feat_names].copy()
    sub = sub.sort_values(["label", "group"], ascending=[False, True])

    X      = sub[feat_names].values.astype(float)
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    y_labels = [f"{'R' if r.label==1 else 'G'} {r.series} [{r.group}]"
                for r in sub.itertuples()]
    hover = [
        [f"<b>{feat_names[j]}</b><br>Raw: {X[i,j]:.3f}<br>Z: {X_norm[i,j]:.2f}"
         for j in range(len(feat_names))]
        for i in range(len(y_labels))
    ]

    fig = go.Figure(go.Heatmap(
        z=X_norm, x=feat_names, y=y_labels,
        colorscale="RdBu", zmid=0,
        text=hover, hoverinfo="text",
        colorbar=dict(title="Z-score"),
        xgap=2, ygap=1,
    ))
    fig.update_layout(
        title=dict(text="Feature Heatmap — All Series (interactive)", font=dict(size=16)),
        xaxis=dict(title="Feature", tickangle=-30),
        yaxis=dict(title="Series"),
        height=max(500, len(y_labels) * 32),
        template="plotly_dark",
        margin=dict(l=260, r=60, t=80, b=100),
    )
    fig.write_html(out_path)
    print(f"  Saved: {out_path}")


def plot_spatial_heatmaps(df, feat_names):
    """לכל פיצ'ר: גריד של פרוסות CT עם overlay של הפיצ'ר."""
    series_list = df["series"].tolist()
    labels      = df["label"].tolist()

    n_series = len(series_list)
    n_cols   = min(5, n_series)
    n_rows   = (n_series + n_cols - 1) // n_cols

    for feat in feat_names:
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 3.2, n_rows * 3.4))
        axes = np.array(axes).flatten()
        fig.suptitle(f"Spatial heatmap -- {feat}",
                     fontsize=13, fontweight="bold", y=1.01)

        for idx, (stem, label) in enumerate(zip(series_list, labels)):
            ax = axes[idx]
            ct_sl, msk_sl = load_best_axial_slice(stem)
            if ct_sl is None:
                ax.axis("off")
                continue

            ct_disp = np.clip(ct_sl, -100, 400)
            ct_disp = (ct_disp - ct_disp.min()) / (ct_disp.max() - ct_disp.min() + 1e-10)
            ax.imshow(ct_disp, cmap="gray", origin="lower", aspect="auto")

            fmap = sliding_feature_map(ct_sl, msk_sl, feat)
            fmap_masked = np.ma.masked_where(msk_sl == 0, fmap)
            im = ax.imshow(fmap_masked, cmap="hot", alpha=0.72,
                           origin="lower", aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

            c = COLOR_CANCER if label == 1 else COLOR_HEALTHY
            ax.set_title(f"{stem.replace('_', ' ')}\n[{'Cancer' if label==1 else 'Healthy'}]",
                         fontsize=6.5, color=c, fontweight="bold", pad=3)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(c)
                spine.set_linewidth(2)

        for i in range(n_series, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"spatial_{feat}.png")
        plt.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")
