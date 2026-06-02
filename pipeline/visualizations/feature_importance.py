"""
גרף עמודות: F-score ו-ICC לכל פיצ'ר.
צבע לפי שיטת בחירה: VaRFS בלבד / ANOVA בלבד / שניהם / לא נבחר.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_feature_importance(fnames, f_scores, icc_vals, varfs_idx, anova_idx, out_path):
    varfs_set = set(varfs_idx)
    anova_set = set(anova_idx)
    colors = []
    for i in range(len(fnames)):
        if i in varfs_set and i in anova_set: colors.append("#FF6F00")
        elif i in varfs_set:                  colors.append("#1565C0")
        elif i in anova_set:                  colors.append("#2E7D32")
        else:                                 colors.append("#9E9E9E")

    sort_idx = np.argsort(f_scores)[::-1]
    snames   = [fnames[i] for i in sort_idx]
    sscores  = f_scores[sort_idx]
    sicc     = icc_vals[sort_idx]
    scols    = [colors[i] for i in sort_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.barh(snames, sscores, color=scols); ax1.set_xlabel("F-score"); ax1.invert_yaxis()
    ax2.barh(snames, sicc,    color=scols); ax2.set_xlabel("ICC");     ax2.invert_yaxis()

    fig.legend(handles=[
        mpatches.Patch(color="#FF6F00", label="Both"),
        mpatches.Patch(color="#1565C0", label="VaRFS only"),
        mpatches.Patch(color="#2E7D32", label="ANOVA only"),
        mpatches.Patch(color="#9E9E9E", label="Not selected"),
    ], loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Feature Importance: F-score vs ICC", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
