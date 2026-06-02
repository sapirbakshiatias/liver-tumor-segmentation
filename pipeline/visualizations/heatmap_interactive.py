"""
Plotly heatmap אינטרקטיבי עם hover info (נשמר כ-HTML).
"""
import numpy as np
import plotly.graph_objects as go


def plot_heatmap_interactive(df, feat_names, out_path):
    sub = df[["series", "group", "label"] + feat_names].copy()
    sub = sub.sort_values(["label", "group"], ascending=[False, True])

    X      = sub[feat_names].values.astype(float)
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    y_labels = [f"{'R' if r.label==1 else 'G'} {r.series}" for r in sub.itertuples()]
    hover    = [
        [f"<b>{feat_names[j]}</b><br>Raw: {X[i,j]:.3f}<br>Z: {X_norm[i,j]:.2f}"
         for j in range(len(feat_names))]
        for i in range(len(y_labels))
    ]

    fig = go.Figure(go.Heatmap(
        z=X_norm, x=feat_names, y=y_labels,
        colorscale="RdBu", zmid=0,
        text=hover, hoverinfo="text",
        colorbar=dict(title="Z-score"), xgap=2, ygap=1,
    ))
    fig.update_layout(
        title="Feature Heatmap (interactive)",
        xaxis=dict(tickangle=-30),
        height=max(500, len(y_labels)*32),
        template="plotly_dark",
        margin=dict(l=260, r=60, t=80, b=100),
    )
    fig.write_html(out_path)
    print(f"  Saved: {out_path}")
