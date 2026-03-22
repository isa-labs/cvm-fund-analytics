"""
Interactive visualization using Plotly
Used by app.py (Streamlit)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = px.colors.qualitative.Bold

def plot_sharpe_bar(
    metrics_df: pd.DataFrame,
    n: int = 15,
    title: str = "Top Funds by Sharpe Ratio",
) -> go.Figure:
    # Horizontal bar chart of top n funds ranked by Sharpe ratio
    df = (
        metrics_df[["DENOM_SOCIAL", "sharpe_ratio"]]
        .dropna()
        .sort_values("sharpe_ratio", ascending=False)
        .head(n)
        .sort_values("sharpe_ratio")
    )

    fig = px.bar(
        df,
        x="sharpe_ratio",
        y="DENOM_SOCIAL",
        orientation="h",
        title=title,
        labels={"sharpe_ratio": "Sharpe Ratio", "DENOM_SOCIAL": ""},
        text=df["sharpe_ratio"].round(2),
        color="sharpe_ratio",
        color_continuous_scale="Blues",
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=500,
        coloraxis_showscale=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="IBM Plex Sans, sans-serif", color="#0f172a"),
        title_font=dict(size=14, family="IBM Plex Mono, monospace", color="#0f172a"),
        xaxis=dict(tickfont=dict(color="#0f172a")),
        yaxis=dict(tickfont=dict(size=10, color="#0f172a")),
        margin=dict(l=10, r=80, t=50, b=20),
    )
    fig.update_xaxes(title_font_color="#0f172a", tickfont_color="#0f172a")
    fig.update_yaxes(title_font_color="#0f172a", tickfont_color="#0f172a")
    return fig

def plot_risk_return_scatter(
    metrics_df: pd.DataFrame,
    title: str = "Risk vs Return",
) -> go.Figure:
    """
    Scatter plot of annualized volatility (x) vs annualized return (y),
    with bubble size and color proportional to Sharpe ratio
    """
    df = metrics_df[
        ["DENOM_SOCIAL", "annualized_return", "annualized_volatility", "sharpe_ratio"]
    ].dropna().copy()

    df["ann_return_pct"] = df["annualized_return"] * 100
    df["ann_vol_pct"] = df["annualized_volatility"] * 100
    df["size"] = (df["sharpe_ratio"].clip(lower=0) * 10 + 5).clip(upper=40)

    fig = px.scatter(
        df,
        x="ann_vol_pct",
        y="ann_return_pct",
        color="sharpe_ratio",
        size="size",
        hover_name="DENOM_SOCIAL",
        hover_data={
            "ann_return_pct": ":.1f",
            "ann_vol_pct": ":.1f",
            "sharpe_ratio": ":.2f",
            "size": False,
        },
        labels={
            "ann_vol_pct": "Annualized Volatility (%)",
            "ann_return_pct": "Annualized Return (%)",
            "sharpe_ratio": "Sharpe Ratio",
        },
        color_continuous_scale="RdYlGn",
        title=title,
    )

    fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8", line_width=1)
    fig.update_layout(
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="IBM Plex Sans, sans-serif", color="#0f172a"),
        title_font=dict(size=14, family="IBM Plex Mono, monospace", color="#0f172a"),
        xaxis=dict(tickfont=dict(color="#0f172a")),
        yaxis=dict(tickfont=dict(color="#0f172a")),
        margin=dict(l=10, r=10, t=50, b=20),
    )
    fig.update_coloraxes(colorbar_tickfont_color="#0f172a", colorbar_title_font_color="#0f172a")
    fig.update_xaxes(title_font_color="#0f172a", tickfont_color="#0f172a")
    fig.update_yaxes(title_font_color="#0f172a", tickfont_color="#0f172a")
    return fig

def plot_cumulative_returns(
    daily_data: pd.DataFrame,
    cnpjs: list[str],
    labels: dict[str, str] | None = None,
    title: str = "Cumulative Return Comparison",
) -> go.Figure:
    # Line chart of cumulative returns for a list of funds.
    fig = go.Figure()

    for i, cnpj in enumerate(cnpjs):
        subset = (
            daily_data[daily_data["CNPJ_FUNDO_CLASSE"] == cnpj]
            .sort_values("DT_COMPTC")
            .set_index("DT_COMPTC")["VL_QUOTA"]
            .dropna()
        )
        if subset.empty:
            continue

        cumret = (subset / subset.iloc[0] - 1) * 100

        if cumret.empty:
            continue

        label = labels.get(cnpj, cnpj) if labels else cnpj
        # Truncate long names
        label = label[:50] + "..." if len(label) > 50 else label

        fig.add_trace(go.Scatter(
            x=cumret.index,
            y=cumret.values,
            name=label,
            mode="lines",
            line=dict(width=2, color=COLORS[i % len(COLORS)]),
            hovertemplate="%{fullData.name}<br>%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8", line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        height=450,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="IBM Plex Sans, sans-serif", color="#0f172a"),
        title_font=dict(size=14, family="IBM Plex Mono, monospace", color="#0f172a"),
        xaxis=dict(tickfont=dict(color="#0f172a")),
        yaxis=dict(tickfont=dict(color="#0f172a")),
        legend=dict(orientation="v", x=1.01, y=1),
        margin=dict(l=10, r=10, t=50, b=20),
        hovermode="x unified",
    )
    fig.update_xaxes(title_font_color="#0f172a", tickfont_color="#0f172a")
    fig.update_yaxes(title_font_color="#0f172a", tickfont_color="#0f172a")
    return fig

def plot_elbow(elbow_df: pd.DataFrame) -> go.Figure:
    # Plots inertia and silhouette score vs. k side by side
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Elbow - Inertia", "Silhouette Score"),
    )

    fig.add_trace(go.Scatter(
        x=elbow_df["k"], y=elbow_df["inertia"],
        mode="lines+markers",
        marker=dict(color="#2563EB", size=8),
        line=dict(color="#2563EB", width=2),
        name="Inertia",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=elbow_df["k"], y=elbow_df["silhouette"],
        mode="lines+markers",
        marker=dict(color="#16A34A", size=8),
        line=dict(color="#16A34A", width=2),
        name="Silhouette",
    ), row=1, col=2)

    fig.update_layout(
        title="Optimal Number of Clusters",
        height=380,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="IBM Plex Sans, sans-serif", color="#0f172a"),
        title_font=dict(size=14, family="IBM Plex Mono, monospace", color="#0f172a"),
        xaxis=dict(tickfont=dict(color="#0f172a")),
        yaxis=dict(tickfont=dict(color="#0f172a")),
        showlegend=False,
        margin=dict(l=10, r=10, t=70, b=20),
    )
    fig.update_xaxes(title_font_color="#0f172a", tickfont_color="#0f172a")
    fig.update_yaxes(title_font_color="#0f172a", tickfont_color="#0f172a")
    return fig

def plot_pca_clusters(
    metrics_df: pd.DataFrame,
    clustered_df: pd.DataFrame,
    title: str = "Fund Clusters - PCA Projection",
) -> go.Figure:
    # PCA 2D scatter colored by cluster
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from src.clustering import prepare_features, FEATURE_COLS

    X_scaled, _ = prepare_features(metrics_df)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100

    labels = clustered_df.loc[X_scaled.index, "cluster_label"].fillna("Unknown")
    names = clustered_df.loc[X_scaled.index, "DENOM_SOCIAL"].fillna("") if "DENOM_SOCIAL" in clustered_df.columns else pd.Series([""] * len(X_scaled), index=X_scaled.index)

    plot_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "cluster_label": labels.values,
        "name": names.values,
    })

    fig = px.scatter(
        plot_df,
        x="PC1", y="PC2",
        color="cluster_label",
        hover_name="name",
        hover_data={"PC1": ":.2f", "PC2": ":.2f"},
        labels={
            "PC1": f"PC1 ({var[0]:.1f}% variance)",
            "PC2": f"PC2 ({var[1]:.1f}% variance)",
            "cluster_label": "Cluster",
        },
        color_discrete_sequence=COLORS,
        title=title,
        opacity=0.65,
    )

    fig.update_traces(marker=dict(size=6))
    fig.update_layout(
        height=450,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="IBM Plex Sans, sans-serif", color="#0f172a"),
        title_font=dict(size=14, family="IBM Plex Mono, monospace", color="#0f172a"),
        xaxis=dict(tickfont=dict(color="#0f172a")),
        yaxis=dict(tickfont=dict(color="#0f172a")),
        margin=dict(l=10, r=10, t=50, b=20),
    )
    fig.update_layout(legend=dict(font=dict(color="#0f172a"), title_font=dict(color="#0f172a")))
    fig.update_xaxes(title_font_color="#0f172a", tickfont_color="#0f172a")
    fig.update_yaxes(title_font_color="#0f172a", tickfont_color="#0f172a")
    return fig

def plot_cluster_profiles(
    clustered_df: pd.DataFrame,
) -> go.Figure:
    # Horizontal bar chart comparing mean metrics per cluster
    from src.clustering import cluster_summary

    summary = cluster_summary(clustered_df).reset_index()

    metrics_to_plot = [
        ("annualized_return", "Ann. Return", True),
        ("annualized_volatility", "Ann. Volatility", True),
        ("sharpe_ratio", "Sharpe Ratio", False),
    ]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[m[1] for m in metrics_to_plot],
    )

    colors = ["#16A34A", "#DC2626", "#2563EB"]

    for col_idx, (metric, label, is_pct) in enumerate(metrics_to_plot, start=1):
        values = summary[metric] * (100 if is_pct else 1)
        text = [f"{v:.1f}%" if is_pct else f"{v:.2f}" for v in values]

        fig.add_trace(go.Bar(
            x=values,
            y=summary["cluster_label"],
            orientation="h",
            text=text,
            textposition="outside",
            marker_color=colors[col_idx - 1],
            showlegend=False,
            hovertemplate="%{y}: %{text}<extra></extra>",
        ), row=1, col=col_idx)

    fig.update_layout(
        title="Cluster Profiles - Mean Metrics",
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="IBM Plex Sans, sans-serif", color="#0f172a"),
        title_font=dict(size=14, family="IBM Plex Mono, monospace", color="#0f172a"),
        xaxis=dict(tickfont=dict(color="#0f172a")),
        yaxis=dict(tickfont=dict(color="#0f172a")),
        margin=dict(l=10, r=80, t=70, b=20),
    )
    fig.update_xaxes(title_font_color="#0f172a", tickfont_color="#0f172a")
    fig.update_yaxes(title_font_color="#0f172a", tickfont_color="#0f172a")
    return fig