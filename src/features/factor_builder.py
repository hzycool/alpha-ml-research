"""Price-volume alpha factor and forward-label construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.factor_dictionary import get_factor_list


def _rolling_corr(df: pd.DataFrame, x: str, y: str, window: int, min_periods: int | None = None) -> pd.Series:
    """Compute rolling correlation within each stock."""
    min_periods = min_periods or max(5, window // 2)
    return df.groupby("code", group_keys=False).apply(
        lambda g: g[x].rolling(window, min_periods=min_periods).corr(g[y]),
        include_groups=False,
    )


def build_alpha_factors(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Construct daily price-volume alpha factors using only current and past data."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["code", "date"]).reset_index(drop=True)
    g = out.groupby("code", group_keys=False)
    out["daily_ret"] = g["close"].pct_change()

    for n in [1, 3, 5, 10, 20, 60, 120]:
        out[f"ret_{n}"] = g["close"].pct_change(n)

    for n in [5, 10, 20, 60]:
        out[f"volatility_{n}"] = g["daily_ret"].transform(lambda s: s.rolling(n, min_periods=max(3, n // 2)).std())

    for n in [5, 10, 20]:
        high_roll = g["high"].transform(lambda s: s.rolling(n, min_periods=max(3, n // 2)).max())
        low_roll = g["low"].transform(lambda s: s.rolling(n, min_periods=max(3, n // 2)).min())
        out[f"amplitude_{n}"] = high_roll / low_roll - 1

    for n in [5, 20, 60]:
        out[f"volume_mean_{n}"] = g["volume"].transform(lambda s: s.rolling(n, min_periods=max(3, n // 2)).mean())
        out[f"amount_mean_{n}"] = g["amount"].transform(lambda s: s.rolling(n, min_periods=max(3, n // 2)).mean())
        if "turnover_rate" in out.columns:
            out[f"turnover_mean_{n}"] = g["turnover_rate"].transform(lambda s: s.rolling(n, min_periods=max(3, n // 2)).mean())
        else:
            out[f"turnover_mean_{n}"] = np.nan

    out["corr_ret_volume_10"] = _rolling_corr(out, "daily_ret", "volume", 10)
    out["corr_ret_volume_20"] = _rolling_corr(out, "daily_ret", "volume", 20)
    out["corr_close_volume_20"] = _rolling_corr(out, "close", "volume", 20)
    out["corr_ret_amount_20"] = _rolling_corr(out, "daily_ret", "amount", 20)

    out["volume_chg_5_20"] = out["volume_mean_5"] / out["volume_mean_20"] - 1
    out["amount_chg_5_20"] = out["amount_mean_5"] / out["amount_mean_20"] - 1
    out["turnover_chg_5_20"] = out["turnover_mean_5"] / out["turnover_mean_20"] - 1

    high_20 = g["high"].transform(lambda s: s.rolling(20, min_periods=10).max())
    low_20 = g["low"].transform(lambda s: s.rolling(20, min_periods=10).min())
    high_60 = g["high"].transform(lambda s: s.rolling(60, min_periods=30).max())
    low_60 = g["low"].transform(lambda s: s.rolling(60, min_periods=30).min())
    out["close_to_high_20"] = out["close"] / high_20 - 1
    out["close_to_low_20"] = out["close"] / low_20 - 1
    out["close_position_20"] = (out["close"] - low_20) / (high_20 - low_20)
    out["close_position_60"] = (out["close"] - low_60) / (high_60 - low_60)
    out["ret_20_div_vol_20"] = out["ret_20"] / out["volatility_20"]
    out["ret_60_div_vol_60"] = out["ret_60"] / out["volatility_60"]

    factor_cols = get_factor_list()
    out[factor_cols] = out[factor_cols].replace([np.inf, -np.inf], np.nan)
    return out.sort_values(["date", "code"]).reset_index(drop=True), factor_cols


def add_forward_returns(df: pd.DataFrame, horizons: list[int], price_col_preference: str = "open") -> pd.DataFrame:
    """Add forward return labels from t+1 to t+h+1 to avoid look-ahead bias."""
    out = df.copy().sort_values(["code", "date"]).reset_index(drop=True)
    price_col = price_col_preference if price_col_preference in out.columns else "close"
    g = out.groupby("code", group_keys=False)
    for h in horizons:
        out[f"y_{h}d"] = g[price_col].shift(-(h + 1)) / g[price_col].shift(-1) - 1
    return out.sort_values(["date", "code"]).reset_index(drop=True)
