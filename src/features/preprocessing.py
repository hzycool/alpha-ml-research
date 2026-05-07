"""Cross-sectional factor winsorization, standardization, and missing handling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.neutralization import neutralize_by_market_cap


def _winsorize_frame(frame: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    """Winsorize columns by quantiles within a date cross-section."""
    q_low = frame.quantile(lower)
    q_high = frame.quantile(upper)
    return frame.clip(lower=q_low, upper=q_high, axis=1)


def _zscore_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score for a date cross-section."""
    mean = frame.mean()
    std = frame.std(ddof=0).replace(0, np.nan)
    return (frame - mean) / std


def preprocess_factors(
    df: pd.DataFrame,
    factor_cols: list[str],
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
    neutralize_market_cap: bool = True,
) -> pd.DataFrame:
    """Apply date-wise winsorization, optional cap neutralization, and z-score.

    Missing factor values remain missing; model training drops rows without a
    complete feature vector and label instead of filling everything with zero.
    """
    out = df.copy()
    out[factor_cols] = out[factor_cols].replace([np.inf, -np.inf], np.nan)
    out[factor_cols] = out.groupby("date", group_keys=False)[factor_cols].apply(
        lambda x: _winsorize_frame(x, winsor_lower, winsor_upper)
    )
    if neutralize_market_cap and "market_cap" in out.columns:
        out = neutralize_by_market_cap(out, factor_cols)
        out[factor_cols] = out.groupby("date", group_keys=False)[factor_cols].apply(
            lambda x: _winsorize_frame(x, winsor_lower, winsor_upper)
        )
    out[factor_cols] = out.groupby("date", group_keys=False)[factor_cols].apply(_zscore_frame)
    out[factor_cols] = out[factor_cols].replace([np.inf, -np.inf], np.nan)
    return out


def drop_model_missing(df: pd.DataFrame, factor_cols: list[str], label_col: str) -> pd.DataFrame:
    """Drop samples missing any model feature or the target label."""
    needed = ["date", "code", label_col, *factor_cols]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required model columns: {missing}")
    return df.dropna(subset=[label_col, *factor_cols]).copy()
