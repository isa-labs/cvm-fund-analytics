"""
Visualization helpers for the CVM Fund Screener
Generates charts using Plotly (interactive) and Matplotlib (static/export)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Color palette
COLORS = [
    "#2563EB", "#16A34A", "#DC2626", "#D97706",
    "#7C3AED", "#0891B2", "#DB2777", "#65A30D",
]

def plot_cumulative_returns(
    daily_data: pd.DataFrame,
    cnpjs: list[str],
    labels: dict[str, str] | None = None,
    title: str = "Cumulative Return Comparison",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Plots cumulative return curves for a list of funds
    daily_data: DataFrame with [CNPJ_FUNDO_CLASSE, DT_COMPTC, VL_QUOTA]
    cnpjs: list of CNPJ strings to plot
    labels: optional dict mapping CNPJ → display name
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

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
        
        # if cumret.min() < -0.999:
        #     continue
        
        label = labels.get(cnpj, cnpj) if labels else cnpj
        color = COLORS[i % len(COLORS)]
        ax.plot(cumret.index, cumret.values, label=label, color=color, linewidth=1.8)

    ax.axhline(0, color="#94A3B8", linewidth=0.8, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

def plot_sharpe_bar(
    metrics_df: pd.DataFrame,
    n: int = 15,
    title: str = "Top Funds by Sharpe Ratio",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Horizontal bar chart of top n funds ranked by Sharpe ratio.
    Expects metrics_df to have columns [DENOM_SOCIAL, sharpe_ratio]
    """
    df = (
        metrics_df[["DENOM_SOCIAL", "sharpe_ratio"]]
        .dropna()
        .sort_values("sharpe_ratio", ascending=False)
        .head(n)
        .sort_values("sharpe_ratio")          # ascending for horizontal bar
    )

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    bars = ax.barh(
        df["DENOM_SOCIAL"].str[:40],           # truncate long names
        df["sharpe_ratio"],
        color="#2563EB",
        edgecolor="none",
        height=0.6,
    )

    for bar, val in zip(bars, df["sharpe_ratio"]):
        ax.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=8, color="#334155"
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Sharpe Ratio")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    fig.tight_layout()
    return fig

def plot_risk_return_scatter(
    metrics_df: pd.DataFrame,
    title: str = "Risk vs. Return",
    figsize: tuple = (9, 7),
) -> plt.Figure:
    """
    Scatter plot of annualized volatility (x) vs annualized return (y),
    with bubble size proportional to |Sharpe ratio|
    """
    df = metrics_df[
        ["DENOM_SOCIAL", "annualized_return", "annualized_volatility", "sharpe_ratio"]
    ].dropna()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    sizes = (df["sharpe_ratio"].clip(lower=0) * 60).clip(lower=10)

    sc = ax.scatter(
        df["annualized_volatility"] * 100,
        df["annualized_return"] * 100,
        s=sizes,
        c=df["sharpe_ratio"],
        cmap="RdYlGn",
        alpha=0.7,
        edgecolors="#CBD5E1",
        linewidths=0.4,
    )

    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Return (%)")
    ax.axhline(0, color="#94A3B8", linewidth=0.7, linestyle="--")
    ax.grid(color="#E2E8F0", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig