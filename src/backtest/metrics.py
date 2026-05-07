"""Portfolio performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(nav: pd.Series) -> float:
    """Compute maximum drawdown from a NAV series."""
    nav = nav.dropna()
    if nav.empty:
        return np.nan
    drawdown = nav / nav.cummax() - 1
    return float(drawdown.min())


def performance_metrics(
    returns: pd.Series,
    nav: pd.Series,
    turnover: pd.Series | None = None,
    periods_per_year: float = 252 / 5,
) -> dict[str, float]:
    """Compute annualized return, volatility, Sharpe, drawdown, and turnover."""
    returns = returns.dropna()
    if returns.empty:
        return {
            "total_return": np.nan,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "turnover": np.nan,
        }
    total_return = float(nav.dropna().iloc[-1] - 1)
    annual_return = float((1 + total_return) ** (periods_per_year / len(returns)) - 1)
    annual_volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year)) if returns.std(ddof=1) != 0 else np.nan
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(nav),
        "turnover": float(turnover.mean()) if turnover is not None and len(turnover) else np.nan,
    }
