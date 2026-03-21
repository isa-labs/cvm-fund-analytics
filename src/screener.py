"""
Filters and ranks funds based on computed metrics
Usage example:
    from src.screener import Screener
    screener = Screener(metrics_df, register_df)
    top = screener.filter(fund_class="FIA").rank_by("sharpe_ratio").top(20)
"""

import pandas as pd

# CVM fund class codes (CLASSE column in cadastro)
FUND_CLASSES = {
    "Ações": "Fundo de Investimento em Ações",
    "Multimercado": "Fundo de Investimento Multimercado",
    "Renda Fixa": "Fundo de Investimento Renda Fixa",
    "Referenciado": "Fundo de Investimento Referenciado",
    "FIDC": "Fundo de Investimento em Direitos Creditórios",
    "FII": "Fundo de Investimento Imobiliário",
    "Cambial": "Fundo de Investimento Cambial",
    "Curto Prazo": "Fundo de Investimento Curto Prazo",
}

class Screener:
    def __init__(self, metrics: pd.DataFrame, register: pd.DataFrame):
        """
        metrics: output of metrics.build_metrics_table()
        register: output of ingest.load_register()
        """
        register_clean = register[["CNPJ_FUNDO", "DENOM_SOCIAL", "CLASSE", "SIT"]].copy()
        register_clean = register_clean.rename(columns={"CNPJ_FUNDO": "CNPJ_BASE"})
        register_clean = register_clean.sort_values("SIT").drop_duplicates(subset="CNPJ_BASE", keep="first")
        register_clean = register_clean.set_index("CNPJ_BASE")

        self.df = metrics.join(register_clean, how="left")
        self._filtered = self.df.copy()

    def filter(
        self,
        fund_class: str | None = None,
        active_only: bool = True,
        min_sharpe: float | None = None,
        max_drawdown: float | None = None,
    ) -> "Screener":
        """
        Applies filters in-place on the internal working DataFrame
        fund_class: CVM class string ("Ações", "Multimercado")
        active_only: keep only funds with SIT == "EM FUNCIONAMENTO NORMAL"
        min_sharpe: minimum Sharpe ratio threshold
        max_drawdown: maximum allowed drawdown (-0.10 means -10%)
        """
        df = self.df.copy()

        if active_only:
            df = df[df["SIT"] == "EM FUNCIONAMENTO NORMAL"]

        if fund_class:
            df = df[df["CLASSE"].str.contains(fund_class, case=False, na=False)]

        if min_sharpe is not None:
            df = df[df["sharpe_ratio"] >= min_sharpe]

        if max_drawdown is not None:
            df = df[df["max_drawdown"] >= max_drawdown]

        self._filtered = df
        return self

    def rank_by(self, metric: str = "sharpe_ratio", ascending: bool = False) -> "Screener":
        # Sorts the filtered DataFrame by a given metric
        self._filtered = self._filtered.sort_values(metric, ascending=ascending)
        return self

    def top(self, n: int = 10) -> pd.DataFrame:
        # Returns the top n funds after filtering and ranking
        cols = [
            "DENOM_SOCIAL",
            "CLASSE",
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
            "n_days",
        ]
        available = [c for c in cols if c in self._filtered.columns]
        return self._filtered[available].head(n)

    def summary(self) -> pd.DataFrame:
        # Returns descriptive statistics of the filtered universe
        numeric_cols = [
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
        ]
        available = [c for c in numeric_cols if c in self._filtered.columns]
        return self._filtered[available].describe()