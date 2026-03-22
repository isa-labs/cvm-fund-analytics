"""
Streamlit dashboard for CVM Fund Analytics
Run with: streamlit run app.py
"""

import os
import sys
import glob

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingest import load_register
from src.database import get_or_load
from src.metrics import build_metrics_table
from src.screener import Screener
from src.clustering import (
    assign_clusters,
    prepare_features,
    find_optimal_k,
    DEFAULT_LABELS,
)
from src.viz_plotly import (
    plot_sharpe_bar,
    plot_risk_return_scatter,
    plot_cumulative_returns,
    plot_elbow,
    plot_pca_clusters,
    plot_cluster_profiles,
)

# Page config
st.set_page_config(
    page_title="CVM Fund Analytics",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
    .stMetric label { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #64748b; }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("CVM Fund Analytics")
st.caption("Risk/return analysis and clustering of Brazilian investment funds. Public CVM data.")
st.divider()

# Sidebar filters
with st.sidebar:
    st.header("Filters")

    # Detect available months from disk
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    available_files = sorted(glob.glob(os.path.join(BASE_DIR, "data/raw/inf_diario_fi_*.csv")))
    available_months = []
    for f in available_files:
        base = os.path.basename(f)
        ym = base.replace("inf_diario_fi_", "").replace(".csv", "")
        if len(ym) == 6:
            available_months.append(f"{ym[:4]}-{ym[4:]}")
    
    # Generate all months from 2025-01 to current month as options
    all_months = []
    for year in range(2025, 2027):
        for month in range(1, 13):
            ym = f"{year}-{month:02d}"
            all_months.append(ym)
            if ym == pd.Timestamp.now().strftime("%Y-%m"):
                break
        else:
            continue
        break

    st.markdown("**Period (year-month)**")
    st.caption("✅ = cached on disk · ☁️ = will download from CVM")

    # Show which months are cached
    options_display = []
    for ym in all_months:
        if ym in available_months:
            options_display.append(f"✅ {ym}")
        else:
            options_display.append(f"☁️ {ym}")

    selected_display = st.multiselect(
        "Select months",
        options=options_display,
        default=[f"✅ {ym}" for ym in available_months],
    )
    selected_months = [s.replace("✅ ", "").replace("☁️ ", "") for s in selected_display]

    fund_classes = {
        "Equity": "Ações",
        "Multi-Strategy": "Multimercado",
        "Fixed Income": "Renda Fixa",
        "Index-Tracking": "Referenciado",
        "FIDC": "FIDC",
        "FII": "FII",
        "FX": "Cambial",
        "Short-Term": "Curto Prazo",
    }
    selected_class_label = st.selectbox("Fund class", options=list(fund_classes.keys()), index=0)
    selected_class = fund_classes[selected_class_label]

    min_sharpe = st.slider("Minimum Sharpe ratio", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    top_n = st.slider("Top n funds", min_value=5, max_value=30, value=15, step=1)
    min_days = st.slider("Minimum trading days", min_value=20, max_value=200, value=60, step=10)

# Load data
@st.cache_data(show_spinner="Loading CVM data")
def load_data(months: list[str], _min_days: int):
    from src.database import get_or_load

    daily = get_or_load(list(months))

    if daily is None or daily.empty:
        return None, None, None

    metrics = build_metrics_table(daily, min_days=_min_days)
    register = load_register()
    return daily, metrics, register

@st.cache_data(show_spinner="Joining register and filtering")
def build_metrics_named(_metrics, _register):
    reg_dedup = (
        _register.rename(columns={"CNPJ_FUNDO": "CNPJ_BASE"})
        .sort_values("SIT")
        .drop_duplicates(subset="CNPJ_BASE", keep="first")
        .set_index("CNPJ_BASE")[["DENOM_SOCIAL", "CLASSE", "SIT"]]
    )
    mn = _metrics.join(reg_dedup, how="left").dropna(subset=["CLASSE"])
    mn = mn[
        (mn["annualized_volatility"] < 5) &
        (mn["annualized_return"].between(-2, 10))
    ]
    return mn

if not selected_months:
    st.warning("Select at least one month in the sidebar")
    st.stop()

daily, metrics, register = load_data(tuple(selected_months), min_days)

if daily is None:
    st.error("Could not load data. Check that data/raw/ has CVM CSV files")
    st.stop()

metrics_named = build_metrics_named(metrics, register)

# Summary metrics
filtered = metrics_named[metrics_named["CLASSE"].str.contains(selected_class, na=False)]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Funds analyzed", f"{len(filtered):,}")
col2.metric("Median Sharpe", f"{filtered['sharpe_ratio'].median():.2f}")
col3.metric("Median Ann. Return", f"{filtered['annualized_return'].median():.1%}")
col4.metric("Median Volatility", f"{filtered['annualized_volatility'].median():.1%}")
col5.metric("Median Max Drawdown", f"{filtered['max_drawdown'].median():.1%}")

st.divider()

# Charts
col_left, col_right = st.columns(2)

# Sharpe bar
with col_left:
    st.subheader(f"Top {top_n} Funds - Sharpe Ratio")
    fig1 = plot_sharpe_bar(
        filtered,
        n=top_n,
        title=f"Top {top_n} {selected_class_label} Funds - Sharpe Ratio",
    )
    st.plotly_chart(fig1, width="stretch")

# Risk vs Return scatter
with col_right:
    st.subheader("Risk vs Return")
    fim_clean = filtered[
        (filtered["annualized_volatility"] < 2) &
        (filtered["annualized_return"].between(-1, 5))
    ]
    fig2 = plot_risk_return_scatter(
        fim_clean,
        title=f"Risk vs Return - {selected_class_label}",
    )
    st.plotly_chart(fig2, width="stretch")

st.divider()

# Cumulative return
st.subheader(f"Cumulative Return - Top 5 {selected_class_label} Funds")
screener = Screener(metrics, register)
top5 = (
    screener
    .filter(fund_class=selected_class, active_only=False, min_sharpe=min_sharpe)
    .rank_by("sharpe_ratio")
    .top(5)
)

if top5.empty:
    st.info("No funds found with the current filters")
else:
    top5_cnpjs = top5.index.tolist()
    labels = top5["DENOM_SOCIAL"].to_dict() if "DENOM_SOCIAL" in top5.columns else None
    fig3 = plot_cumulative_returns(
        daily,
        cnpjs=top5_cnpjs,
        labels=labels,
        title=f"Cumulative Return - Top 5 {selected_class_label} Funds",
    )
    st.plotly_chart(fig3, width="stretch")

st.divider()

# Clustering
st.subheader("Fund Clustering")

col_elbow, col_pca = st.columns(2)

with col_elbow:
    st.markdown("**Optimal k - Elbow + Silhouette**")
    X_scaled, _ = prepare_features(metrics_named)
    elbow_df = find_optimal_k(X_scaled, k_range=range(2, 9))
    fig4 = plot_elbow(elbow_df)
    st.plotly_chart(fig4, width="stretch")

with col_pca:
    st.markdown("**PCA Cluster Projection**")
    clustered = assign_clusters(metrics_named, k=4, cluster_labels=DEFAULT_LABELS)
    fig5 = plot_pca_clusters(
        metrics_named, clustered,
        title="Brazilian Funds - PCA Cluster Projection",
    )
    st.plotly_chart(fig5, width="stretch")

st.subheader("Cluster Profiles - Mean Metrics")
fig6 = plot_cluster_profiles(clustered)
st.plotly_chart(fig6, width="stretch")

st.divider()
st.caption("Data source: CVM open data portal - dados.cvm.gov.br - No authentication required")