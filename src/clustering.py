"""
Clusters CVM-registered funds based on their risk/return profile using K-Means
PCA is applied for 2D visualization
Features used:
- annualized_return
- annualized_volatility
- sharpe_ratio
- max_drawdown
- calmar_ratio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Cluster labels (interpretable names assigned after inspection)
DEFAULT_LABELS = {
    0: "Conservative",
    1: "Aggressive Growth",
    2: "Distressed",
    3: "High Volatility",
}

COLORS = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED", "#0891B2"]

FEATURE_COLS = [
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "calmar_ratio",
]

def prepare_features(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Selects and standardizes features for clustering
    Returns:
    - X_scaled: pd.DataFrame (same index as input)
    - scaler: fitted StandardScaler
    """
    available = [c for c in FEATURE_COLS if c in metrics_df.columns]
    X = metrics_df[available].dropna()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        index=X.index,
        columns=available,
    )
    return X_scaled, scaler

def find_optimal_k(X_scaled: pd.DataFrame, k_range: range = range(2, 10)) -> pd.DataFrame:
    """
    Computes inertia and silhouette score for a range of k values
    to help select the optimal number of clusters (elbow method)
    Returns: pd.DataFrame with columns [k, inertia, silhouette]
    """
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if k > 1 else np.nan
        results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(results)

def fit_clusters(X_scaled: pd.DataFrame, k: int = 4) -> tuple[KMeans, np.ndarray]:
    """
    Fits K-Means with k clusters.
    Returns:
    - model: fitted KMeans
    - labels: cluster assignment array
    """
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    return model, labels

def assign_clusters(
    metrics_df: pd.DataFrame,
    k: int = 4,
    cluster_labels: dict[int, str] | None = None,
) -> pd.DataFrame:
    """
    Full pipeline: standardize → cluster → attach results back to metrics_df
    Parameters:
    - metrics_df: output of metrics.build_metrics_table() merged with register
    - k: number of clusters
    - cluster_labels: optional dict mapping cluster int → descriptive name
    Returns:
    - metrics_df with two new columns: [cluster, cluster_label]
    """
    X_scaled, _ = prepare_features(metrics_df)
    _, labels = fit_clusters(X_scaled, k=k)

    result = metrics_df.copy()
    result.loc[X_scaled.index, "cluster"] = labels.astype(int)

    if cluster_labels is None:
        cluster_labels = {i: f"Cluster {i}" for i in range(k)}
    result["cluster_label"] = result["cluster"].map(cluster_labels)
    return result

def cluster_summary(clustered_df: pd.DataFrame) -> pd.DataFrame:
    # Returns mean metrics per cluster, sorted by Sharpe ratio descending
    numeric = [c for c in FEATURE_COLS if c in clustered_df.columns]
    summary = (
        clustered_df.groupby("cluster_label")[numeric]
        .mean()
        .sort_values("sharpe_ratio", ascending=False)
    )
    summary["count"] = clustered_df.groupby("cluster_label").size()
    return summary

# Visualization
def plot_elbow(elbow_df: pd.DataFrame, figsize: tuple = (10, 4)) -> plt.Figure:
    # Plots inertia and silhouette score vs k side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#F8FAFC")

    for ax in (ax1, ax2):
        ax.set_facecolor("#F8FAFC")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.6)

    ax1.plot(elbow_df["k"], elbow_df["inertia"], marker="o", color="#2563EB", linewidth=2)
    ax1.set_title("Elbow — Inertia", fontweight="bold")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")

    ax2.plot(elbow_df["k"], elbow_df["silhouette"], marker="o", color="#16A34A", linewidth=2)
    ax2.set_title("Silhouette Score", fontweight="bold")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Score")

    fig.suptitle("Optimal Number of Clusters", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig

def plot_pca_clusters(
    metrics_df: pd.DataFrame,
    clustered_df: pd.DataFrame,
    title: str = "Fund Clusters - PCA Projection",
    figsize: tuple = (9, 7),
) -> plt.Figure:
    # Projects funds into 2D using PCA and colors them by cluster
    X_scaled, _ = prepare_features(metrics_df)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    var_explained = pca.explained_variance_ratio_ * 100

    # Align cluster labels with the scaled index
    labels = clustered_df.loc[X_scaled.index, "cluster_label"].fillna("Unknown")
    unique_labels = labels.unique()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    for i, lbl in enumerate(sorted(unique_labels)):
        mask = labels == lbl
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=lbl,
            color=COLORS[i % len(COLORS)],
            alpha=0.55,
            s=20,
            edgecolors="none",
        )

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance)", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.legend(loc="best", fontsize=9, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#E2E8F0", linewidth=0.5)
    fig.tight_layout()
    return fig

def plot_cluster_profiles(
    clustered_df: pd.DataFrame,
    figsize: tuple = (11, 5),
) -> plt.Figure:
    # Horizontal bar chart comparing mean Sharpe, volatility and return per cluster
    summary = cluster_summary(clustered_df).reset_index()
    metrics_to_plot = ["annualized_return", "annualized_volatility", "sharpe_ratio"]
    labels_plot = ["Ann. Return", "Ann. Volatility", "Sharpe Ratio"]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor("#F8FAFC")

    for ax, col, label, color in zip(
        axes, metrics_to_plot, labels_plot,
        ["#16A34A", "#DC2626", "#2563EB"]
    ):
        ax.set_facecolor("#F8FAFC")
        values = summary[col] * (100 if "return" in col or "volatility" in col else 1)
        fmt = "{:.1f}%" if "return" in col or "volatility" in col else "{:.2f}"

        bars = ax.barh(summary["cluster_label"], values, color=color, height=0.5, edgecolor="none")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + abs(values.max()) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(val), va="center", fontsize=8
            )
        ax.set_title(label, fontweight="bold", fontsize=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.5)

    fig.suptitle("Cluster Profiles - Mean Metrics", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig