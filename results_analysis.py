"""
Visualize radiomics features and classification results.
Run after extract_radiomics.py and train_svm.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from collections import Counter

CSV_PATH = r"c:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\liver_radiomics.csv"
K_FEATURES = 2

CANCER_COLOR = "#e74c3c"
HEALTHY_COLOR = "#2ecc71"


def get_feature_frequencies(X, y, feature_names):
    loo = LeaveOneOut()
    all_selected = []
    for train_idx, _ in loo.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_train)
        selector = SelectKBest(f_classif, k=K_FEATURES)
        selector.fit(X_s, y_train)
        selected = [feature_names[i] for i in selector.get_support(indices=True)]
        all_selected.extend(selected)
    return Counter(all_selected)


def plot_feature_frequencies(freq, n_folds, ax):
    features = [f for f, _ in freq.most_common()]
    counts = [c for _, c in freq.most_common()]
    colors = [CANCER_COLOR if c == n_folds else "#3498db" if c > n_folds // 2 else "#95a5a6"
              for c in counts]
    bars = ax.barh(features[::-1], counts[::-1], color=colors[::-1])
    ax.set_xlabel("Selected in N folds (out of 5)")
    ax.set_title("Feature Selection Frequency Across LOOCV Folds")
    ax.axvline(x=n_folds, color="gray", linestyle="--", alpha=0.5, label=f"Max ({n_folds})")
    ax.set_xlim(0, n_folds + 0.5)
    for bar, count in zip(bars[::-1], counts):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{count}/{n_folds}", va="center", fontsize=9)


def plot_top_features(df, feature_names, freq, n_top=4, axes=None):
    top_features = [f for f, _ in freq.most_common(n_top)]
    cancer_mask = df["label"] == 1
    healthy_mask = df["label"] == 0

    for i, feat in enumerate(top_features):
        ax = axes[i]
        cancer_vals = df.loc[cancer_mask, feat].values
        healthy_vals = df.loc[healthy_mask, feat].values

        x_cancer = np.random.normal(0, 0.05, len(cancer_vals))
        x_healthy = np.random.normal(1, 0.05, len(healthy_vals))

        ax.scatter(x_cancer, cancer_vals, color=CANCER_COLOR, s=100, zorder=3, label="Cancer")
        ax.scatter(x_healthy, healthy_vals, color=HEALTHY_COLOR, s=100, zorder=3, label="Healthy")

        # Mean lines
        ax.hlines(cancer_vals.mean(), -0.2, 0.2, colors=CANCER_COLOR, linewidth=2, alpha=0.7)
        ax.hlines(healthy_vals.mean(), 0.8, 1.2, colors=HEALTHY_COLOR, linewidth=2, alpha=0.7)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Cancer", "Healthy"])
        ax.set_title(feat, fontsize=9)
        ax.set_ylabel("Value")

    # Add legend to last plot
    cancer_patch = mpatches.Patch(color=CANCER_COLOR, label="Cancer (future)")
    healthy_patch = mpatches.Patch(color=HEALTHY_COLOR, label="Healthy")
    axes[-1].legend(handles=[cancer_patch, healthy_patch], loc="best", fontsize=8)


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run extract_radiomics.py first.")
        exit(1)

    df = pd.read_csv(CSV_PATH)
    feature_cols = [c for c in df.columns if c not in ("patient", "label")]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(int)

    print("Computing feature selection frequencies...")
    freq = get_feature_frequencies(X, y, feature_cols)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Liver Cancer Prediction — Radiomics Analysis\n(3 cancer-future vs 2 healthy, LOOCV)", fontsize=13)

    # Left: feature frequency bar chart
    ax_freq = fig.add_subplot(1, 2, 1)
    plot_feature_frequencies(freq, n_folds=5, ax=ax_freq)

    # Right: top 4 features scatter plots
    top_4 = [f for f, _ in freq.most_common(4)]
    gs_right = fig.add_gridspec(2, 2, left=0.55, right=0.97, hspace=0.45, wspace=0.4)
    axes_scatter = [
        fig.add_subplot(gs_right[0, 0]),
        fig.add_subplot(gs_right[0, 1]),
        fig.add_subplot(gs_right[1, 0]),
        fig.add_subplot(gs_right[1, 1]),
    ]
    plot_top_features(df, feature_cols, freq, n_top=4, axes=axes_scatter)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    print("Plot displayed.")
