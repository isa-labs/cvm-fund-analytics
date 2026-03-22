# %%
# Source cells (convert with jupytext or paste manually)

# Setup
import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
from src.ingest import download_range, load_register
from src.metrics import build_metrics_table
from src.screener import Screener
from src import plot_cumulative_returns, plot_sharpe_bar, plot_risk_return_scatter
from datetime import datetime

pd.set_option("display.float_format", "{:.4f}".format)
plt.rcParams["figure.dpi"] = 120
# %%
# Download data
# Downloads 6 months of daily CVM fund data (~300k rows)
# On first run this may take 1-2 minutes depending on connection
def load_or_download(start, end):
    start_dt = datetime.strptime(start, "%Y-%m")
    end_dt = datetime.strptime(end, "%Y-%m")
    
    frames = []
    current = start_dt
    while current <= end_dt:
        path = f"data/raw/inf_diario_fi_{current.year}{current.month:02d}.csv"
        if os.path.exists(path):
            print(f"Loading from disk: {path}")
            df = pd.read_csv(path, dtype={"CNPJ_FUNDO_CLASSE": str}, low_memory=False)
            df["CNPJ_BASE"] = df["CNPJ_FUNDO_CLASSE"].str[:18]
            frames.append(df)
        else:
            from src.ingest import download_month
            frames.append(download_month(current.year, current.month))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return pd.concat(frames, ignore_index=True)

daily = load_or_download(start="2025-01", end="2025-12")

register = load_register()

print(f"Daily records: {len(daily):,}")
print(f"Unique funds: {daily['CNPJ_FUNDO_CLASSE'].nunique():,}")
daily.head()
# %%
# Compute metrics
metrics = build_metrics_table(daily, min_days=60)
print(f"Funds with enough history: {len(metrics):,}")
metrics.describe()
# %%
# Screen - top equity funds by Sharpe
screener = Screener(metrics, register)

top_fia = (
    screener
    .filter(fund_class="Ações", active_only=False, min_sharpe=0.0, max_drawdown=-0.40)
    .rank_by("sharpe_ratio")
    .top(20)
)

top_fia[["DENOM_SOCIAL", "annualized_return", "annualized_volatility",
          "sharpe_ratio", "max_drawdown"]].style.format({
    "annualized_return": "{:.1%}",
    "annualized_volatility": "{:.1%}",
    "sharpe_ratio": "{:.2f}",
    "max_drawdown": "{:.1%}",
})
# %%
# Screen - top multi-strategy funds (FIM)
screener2 = Screener(metrics, register)

top_fim = (
    screener2
    .filter(fund_class="Multimercado", active_only=False, min_sharpe=0.0, max_drawdown=-0.30)
    .rank_by("sharpe_ratio")
    .top(20)
)
top_fim.head()
# %%
# Sharpe bar chart
reg_dedup = (
    register.rename(columns={"CNPJ_FUNDO": "CNPJ_BASE"})
    .sort_values("SIT")  # prioriza EM FUNCIONAMENTO NORMAL
    .drop_duplicates(subset="CNPJ_BASE", keep="first")
    .set_index("CNPJ_BASE")[["DENOM_SOCIAL", "CLASSE", "SIT"]]
)

"""
Join computed metrics with fund register to add name, class and status
left join keeps all funds even if not found in the register
"""
metrics_named = metrics.join(reg_dedup, how="left")

# Drop funds with no matching class in the register (unmatched CNPJs)
metrics_named = metrics_named.dropna(subset=["CLASSE"])

"""
Remove funds with unrealistic return or volatility, likely data errors or
funds with very few observations that distort annualized calculations
"""
metrics_named = metrics_named[
    (metrics_named["annualized_volatility"] < 5) & # Cap at 500% annualized vol
    (metrics_named["annualized_return"].between(-2, 10)) # Cap at -200%/+1000% return
]

"""
Remove funds with extreme Sharpe ratios - caused by near-zero volatility
or corrupt NAV series that survived the previous filters
"""
metrics_named = metrics_named[metrics_named["sharpe_ratio"].between(-10, 20)]

fig = plot_sharpe_bar(
    metrics_named[metrics_named["CLASSE"].str.contains("Ações", na=False)],
    n=15,
    title="Top 15 Equity Funds (FIA) - Sharpe Ratio (Jan-Dec 2025)",
)
fig.savefig("outputs/sharpe_bar_fia.png", bbox_inches="tight")
plt.show()
# %%
# Risk-return scatter
fim_clean = metrics_named[
    (metrics_named["CLASSE"].str.contains("Multimercado", na=False)) &
    (metrics_named["annualized_volatility"] < 2) &
    (metrics_named["annualized_return"].between(-1, 5))
]
fig2 = plot_risk_return_scatter(fim_clean, title="Risk vs Return - Multi-Strategy Funds (Multimercado)")
fig2.savefig("outputs/risk_return_fim.png", bbox_inches="tight")
plt.show()
# %%
# Cumulative return - top 5 FIA funds
top5_cnpjs = top_fia.head(5).index.tolist()
labels = top_fia.head(5)["DENOM_SOCIAL"].to_dict()

fig3 = plot_cumulative_returns(
    daily,
    cnpjs=top5_cnpjs,
    labels=labels,
    title="Cumulative Return - Top 5 Equity Funds (FIA)",
)
fig3.savefig("outputs/cumulative_top5_fia.png", bbox_inches="tight")
plt.show()
# %%
# Summary stats
print("=== Universe summary (all active funds, min 60 days) ===\n")
print(screener.summary().to_string())
# %%
# Find optimal k (elbow + silhouette)
from src.clustering import find_optimal_k, prepare_features, assign_clusters
from src.clustering import cluster_summary, plot_elbow, plot_pca_clusters, plot_cluster_profiles

X_scaled, _ = prepare_features(metrics_named)
elbow_df = find_optimal_k(X_scaled, k_range=range(2, 9))

fig4 = plot_elbow(elbow_df)
fig4.savefig("outputs/elbow.png", bbox_inches="tight")
plt.show()
# %%
# Assign clusters (k=4) 
from src.clustering import DEFAULT_LABELS

clustered = assign_clusters(metrics_named, k=4, cluster_labels=DEFAULT_LABELS)
print(clustered["cluster_label"].value_counts())
# %%
# Cluster summary table
summary = cluster_summary(clustered)
summary.style.format({
    "annualized_return": "{:.1%}",
    "annualized_volatility": "{:.1%}",
    "sharpe_ratio": "{:.2f}",
    "max_drawdown": "{:.1%}",
    "calmar_ratio": "{:.2f}",
})
# %%
# PCA scatter by cluster 
fig5 = plot_pca_clusters(metrics_named, clustered, title="Brazilian Funds - PCA Cluster Projection")
fig5.savefig("outputs/pca_clusters.png", bbox_inches="tight")
plt.show()
# %%
# Cluster profiles bar chart
fig6 = plot_cluster_profiles(clustered)
fig6.savefig("outputs/cluster_profiles.png", bbox_inches="tight")
plt.show()