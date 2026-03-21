"""
Computes risk/return metrics for each fund given a daily NAV time series
Metrics:
- Cumulative return
- Annualized return
- Annualized volatility
- Sharpe ratio  (risk-free: CDI proxy = 10.5% p.a., adjustable)
- Maximum drawdown
- Calmar ratio  (annualized return / max drawdown)
"""

import numpy as np
import pandas as pd

RISK_FREE_ANNUAL = 0.105 # CDI approximation, update as needed
TRADING_DAYS = 252

def daily_returns(nav_series: pd.Series) -> pd.Series:
    # Percent change of NAV, dropping the first NaN
    return nav_series.pct_change().dropna()

def cumulative_return(nav_series: pd.Series) -> float:
    # Total return over the full period
    clean = nav_series.dropna()
    if len(clean) < 2:
        return np.nan
    return (clean.iloc[-1] / clean.iloc[0]) - 1

def annualized_return(nav_series: pd.Series) -> float:
    # Geometrically annualized return
    clean = nav_series.dropna()
    n_days = len(clean)
    if n_days < 2:
        return np.nan
    total = cumulative_return(clean)
    return (1 + total) ** (TRADING_DAYS / n_days) - 1

def annualized_volatility(nav_series: pd.Series) -> float:
    # Annualized standard deviation of daily returns
    rets = daily_returns(nav_series)
    if len(rets) < 2:
        return np.nan
    return rets.std() * np.sqrt(TRADING_DAYS)

def sharpe_ratio(nav_series: pd.Series, risk_free: float = RISK_FREE_ANNUAL) -> float:
    # Sharpe ratio = (annualized return - risk free)/annualized volatility
    ann_ret = annualized_return(nav_series)
    ann_vol = annualized_volatility(nav_series)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return (ann_ret - risk_free) / ann_vol

def max_drawdown(nav_series: pd.Series) -> float:
    """
    Maximum peak-to-trough decline
    Returns a negative number (-0.23 means -23%)
    """
    clean = nav_series.dropna()
    if len(clean) < 2:
        return np.nan
    rolling_max = clean.cummax()
    drawdown = (clean - rolling_max) / rolling_max
    return drawdown.min()

def calmar_ratio(nav_series: pd.Series) -> float:
    # Annualized return divided by the absolute max drawdown
    ann_ret = annualized_return(nav_series)
    mdd = max_drawdown(nav_series)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return ann_ret / abs(mdd)

def compute_all(nav_series: pd.Series) -> dict:
    # Convenience function that returns all metrics as a dict
    return {
        "cumulative_return": cumulative_return(nav_series),
        "annualized_return": annualized_return(nav_series),
        "annualized_volatility": annualized_volatility(nav_series),
        "sharpe_ratio": sharpe_ratio(nav_series),
        "max_drawdown": max_drawdown(nav_series),
        "calmar_ratio": calmar_ratio(nav_series),
        "n_days": nav_series.dropna().__len__(),
    }

def build_metrics_table(
    daily_data: pd.DataFrame,
    min_days: int = 60,
) -> pd.DataFrame:
    """
    Receives a DataFrame with columns [CNPJ_FUNDO_CLASSE, DT_COMPTC, VL_QUOTA]
    and returns one row per fund with all metrics
    daily_data: pd.DataFrame
        Output of ingest.download_range(), possibly merged with register
    min_days: int
        Minimum number of trading days required to include a fund
    Returns: pd.DataFrame indexed by CNPJ_FUNDO_CLASSE
    """
    results = []

    for cnpj, group in daily_data.groupby("CNPJ_BASE"):
        series = (
            group.sort_values("DT_COMPTC")
            .set_index("DT_COMPTC")["VL_QUOTA"]
            .dropna()
        )
        
        series = series[series > 0]
        if len(series) < min_days or (series.std() / series.mean()) > 10:
            continue

        row = {"CNPJ_BASE": cnpj}
        row.update(compute_all(series))
        results.append(row)

    df = pd.DataFrame(results).set_index("CNPJ_BASE")
    return df