"""Optional cross-sectional market-cap neutralization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize_by_market_cap(df: pd.DataFrame, factor_cols: list[str], market_cap_col: str = "market_cap") -> pd.DataFrame:
    """Regress each factor on log market capitalization by date and keep residuals."""
    if market_cap_col not in df.columns:
        return df

    out = df.copy()
    log_cap = np.log(out[market_cap_col].replace(0, np.nan))
    out["_log_market_cap"] = log_cap.replace([np.inf, -np.inf], np.nan)

    for date, idx in out.groupby("date").groups.items():
        x = out.loc[idx, "_log_market_cap"]
        x_valid = x.notna()
        if x_valid.sum() < 10:
            continue
        X = np.column_stack([np.ones(x_valid.sum()), x.loc[x_valid].to_numpy()])
        for factor in factor_cols:
            y = out.loc[idx, factor]
            valid = x_valid & y.notna()
            if valid.sum() < 10 or y.loc[valid].std(ddof=0) == 0:
                continue
            Xv = np.column_stack([np.ones(valid.sum()), out.loc[idx, "_log_market_cap"].loc[valid].to_numpy()])
            beta = np.linalg.lstsq(Xv, y.loc[valid].to_numpy(), rcond=None)[0]
            fitted = Xv @ beta
            out.loc[y.loc[valid].index, factor] = y.loc[valid].to_numpy() - fitted

    return out.drop(columns=["_log_market_cap"])
