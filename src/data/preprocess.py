"""Raw panel cleaning and stock-pool filters."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = ["date", "code", "open", "high", "low", "close", "volume", "amount"]


def validate_daily_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize daily OHLCV panel columns."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required daily panel columns: {missing}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["code"] = out["code"].astype(str)
    numeric_cols = [c for c in out.columns if c not in {"date", "code"}]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(["code", "date"]).reset_index(drop=True)


def filter_stock_pool(
    df: pd.DataFrame,
    min_history_days: int = 180,
    min_price: float = 1.0,
    max_missing_ratio: float = 0.25,
) -> pd.DataFrame:
    """Filter suspended, illiquid, and short-history stocks.

    The filter is deliberately simple for a public-data demo. Production
    research should additionally account for ST status, listing age, limit-up
    and limit-down states, and investable universe definitions.
    """
    out = validate_daily_panel(df)
    out = out[(out["close"] >= min_price) & (out["volume"] > 0) & (out["amount"] > 0)]
    keep_codes: list[str] = []
    check_cols = ["open", "high", "low", "close", "volume", "amount"]
    for code, g in out.groupby("code"):
        if len(g) < min_history_days:
            continue
        missing_ratio = g[check_cols].isna().mean().max()
        if missing_ratio <= max_missing_ratio:
            keep_codes.append(code)
    return out[out["code"].isin(keep_codes)].sort_values(["date", "code"]).reset_index(drop=True)
