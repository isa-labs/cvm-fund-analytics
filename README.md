# CVM Fund Analytics

Risk/return analysis and clustering of Brazilian investment funds using public data from the [CVM](https://www.gov.br/cvm/en) (Brazilian Securities and Exchange Commission).

---

## Features

**Risk/Return analysis**
- Downloads CVM monthly NAV data as `.zip` files and consolidates into a single DataFrame
- Persists data locally using **DuckDB**, loads from database on subsequent runs, downloads missing months automatically
- Computes 6 metrics per fund: cumulative return, annualized return, volatility, Sharpe ratio, max drawdown, Calmar ratio
- Screens and ranks funds by class (FIA, FIM, FI-RF, FIC) with chainable filters
- Filters out corrupt NAV series (zero/negative quotes, extreme volatility or return outliers)

**Unsupervised clustering**
- Groups funds by risk/return profile using K-Means
- Elbow method + silhouette score to select optimal k
- PCA projection for 2D cluster visualization
- Cluster profile comparison (mean metrics per group)
- Cluster labels: `Conservative`, `Aggressive Growth`, `High Volatility`, `Distressed`

**Interactive Dashboard**
- Built with Streamlit and Plotly for interactive visualization
- Sidebar filters: period, fund class, minimum Sharpe ratio, top N funds, minimum trading days
- Period selector shows which months are cached locally vs. need to be downloaded
- All 6 charts are interactive with zoom, hover tooltips and dynamic filtering

![dashboard_01](outputs/dashboard_01.png)

![dashboards_02](outputs/dashboard_02.png)

![dashboards_03](outputs/dashboard_03.png)

---

## Metrics

| Metric | Description |
|---|---|
| Cumulative Return | Total NAV appreciation over the period |
| Annualized Return | Geometrically annualized equivalent |
| Annualized Volatility | Std. deviation of daily returns × √252 |
| Sharpe Ratio | Excess return per unit of risk (CDI as risk-free) |
| Maximum Drawdown | Largest peak-to-trough decline |
| Calmar Ratio | Annualized return / \|Max Drawdown\| |

> **Risk-free rate:** CDI approximation (10.5% p.a. adjust `RISK_FREE_ANNUAL` in `metrics.py`)

---

## Structure

```
cvm-fund-analytics/
├── app.py                 # Streamlit interactive dashboard
├── data/
│   ├── raw/               # CVM CSV files (git-ignored)
│   └── cvm.duckdb         # Local DuckDB database (git-ignored)
├── notebooks/
│   └── analysis.py        # End-to-end analysis (jupytext percent format)
├── outputs/               # Saved charts
├── src/
│   ├── __init__.py
│   ├── ingest.py          # CVM data download and register loading
│   ├── metrics.py         # Risk/return metric calculations
│   ├── screener.py        # Fund filtering and ranking
│   ├── clustering.py      # K-Means clustering + PCA visualization
│   ├── database.py        # DuckDB persistence layer
│   ├── viz.py             # Matplotlib charts (notebook/static export)
│   └── viz_plotly.py      # Plotly charts (interactive dashboard)
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/isa-labs/cvm-fund-analytics.git
cd cvm-fund-analytics
pip install -r requirements.txt
```

**Run the interactive dashboard:**

```bash
streamlit run app.py
```

**Run the analysis notebook:**

```bash
jupytext --to notebook notebooks/analysis.py
jupyter lab notebooks/analysis.ipynb
```

**Screening example:**

```python
from src.database import get_or_load
from src.metrics import build_metrics_table
from src.screener import Screener
from src.ingest import load_register

# Loads from DuckDB if available, downloads from CVM otherwise
months = [f"2025-{m:02d}" for m in range(1, 13)]
daily = get_or_load(months)

register = load_register()
metrics = build_metrics_table(daily, min_days=60)

top = (
    Screener(metrics, register)
    .filter(fund_class="Ações", active_only=False, min_sharpe=0.0)
    .rank_by("sharpe_ratio")
    .top(20)
)
```

**Clustering example:**

```python
from src.clustering import assign_clusters, cluster_summary, find_optimal_k
from src.clustering import plot_elbow, plot_pca_clusters, plot_cluster_profiles

# Find optimal k
elbow_df = find_optimal_k(X_scaled, k_range=range(2, 9))

# Assign clusters with interpretable labels
clustered = assign_clusters(metrics_named, k=4)
print(cluster_summary(clustered))

# Visualize
plot_pca_clusters(metrics_named, clustered)
plot_cluster_profiles(clustered)
```

---

## Outputs

Charts are saved to the `outputs/` folder after running the notebook.

**Top 15 Equity Funds - Sharpe Ratio**
![Sharpe Bar](outputs/sharpe_bar_fia.png)

**Risk vs Return - Multi-Strategy Funds**
![Risk Return](outputs/risk_return_fim.png)

**Cumulative Return - Top 5 Equity Funds**
![Cumulative Return](outputs/cumulative_top5_fia.png)

**Optimal Number of Clusters (Elbow + Silhouette)**
![Elbow](outputs/elbow.png)

**PCA Cluster Projection**
![PCA Clusters](outputs/pca_clusters.png)

**Cluster Profiles - Mean Metrics**
![Cluster Profiles](outputs/cluster_profiles.png)

---

## Data source

All data is fetched directly from **CVM's open data portal**. Files are published monthly in `.zip` format, semicolon-separated, Latin-1 encoded. No authentication required.

| Dataset | URL |
|---|---|
| Daily fund NAV (inf_diario) | `dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/` |
| Fund register (cadastro) | `dados.cvm.gov.br/dados/FI/CAD/DADOS/` |

---

## Fund classes

| Class | Description |
|---|---|
| `Ações` | Equity funds (FIA) |
| `Multimercado` | Multi-strategy funds (FIM) |
| `Renda Fixa` | Fixed income funds |
| `FIDC` | Credit rights funds |
| `FIP` | Private equity funds |
| `Referenciado` | Index-tracking funds |
| `FIP Multi` | Multi-strategy private equity funds |
| `FIDC-NP` | Non-performing credit rights funds |
| `FII` | Real estate investment funds |
| `Curto Prazo` | Short-term funds |
| `FIC FIDC` | Fund of credit rights funds |
| `Cambial` | FX funds |
| `Dívida Externa` | External debt funds |
| `FIC FIP` | Fund of private equity funds |
| `FICFIDC-NP` | Fund of non-performing credit rights funds |
| `FMIEE` | Innovative companies investment funds |
| `FIP IE` | Infrastructure private equity funds |
| `FIP EE` | Energy efficiency private equity funds |
| `FIP CS` | Strategic sector private equity funds |
| `FUNCINE` | Film industry investment funds |
| `FMP-FGTS` | FGTS investment funds |
| `FII-FIAGRO` | Agribusiness real estate funds |
| `FIDCFIAGRO` | Agribusiness credit rights funds |
| `FIP PD&I` | R&D private equity funds |
| `FIP-FIAGRO` | Agribusiness private equity funds |
| `FIDC-PIPS` | Social inclusion credit rights funds |

---

## License

MIT - data sourced from CVM under Brazil's [Lei de Acesso à Informação (LAI)](http://www.planalto.gov.br/ccivil_03/_ato2011-2014/2012/lei/l12527.htm).