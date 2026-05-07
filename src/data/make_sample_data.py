"""Generate a small synthetic A-share-like daily panel.

The sample data is designed to exercise the full research pipeline. It is not
real market data and must not be interpreted as investable evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.io import save_parquet


@dataclass(frozen=True)
class SampleDataConfig:
    """Configuration for synthetic panel generation."""

    start_date: str
    end_date: str
    n_stocks: int
    seed: int = 42


def generate_sample_daily_data(config: SampleDataConfig) -> pd.DataFrame:
    """Generate an A-share-like daily OHLCV panel with weak synthetic alpha.

    The process contains a market component, stock-specific idiosyncratic risk,
    weak short-term reversal, and volume/activity effects. These weak signals
    make the pipeline measurable while remaining clearly synthetic.
    """
    rng = np.random.default_rng(config.seed)
    dates = pd.bdate_range(config.start_date, config.end_date)
    n_dates = len(dates)
    market_ret = rng.normal(0.0002, 0.010, n_dates)
    rows: list[dict[str, object]] = []

    for i in range(config.n_stocks):
        code = f"{600000 + i:06d}.SH" if i < config.n_stocks // 2 else f"{300000 + i:06d}.SZ"
        beta = rng.uniform(0.7, 1.3)
        base_price = rng.uniform(8, 80)
        float_shares = rng.uniform(2e8, 3e9)
        base_turnover = rng.uniform(0.004, 0.035)
        close = np.empty(n_dates)
        open_price = np.empty(n_dates)
        high = np.empty(n_dates)
        low = np.empty(n_dates)
        volume = np.empty(n_dates)
        turnover = np.empty(n_dates)
        rets = np.zeros(n_dates)
        activity_state = 0.0
        price = base_price

        for t in range(n_dates):
            prev_5 = rets[max(0, t - 5) : t].sum()
            prev_20 = rets[max(0, t - 20) : t].sum()
            activity_state = 0.92 * activity_state + rng.normal(0, 0.35)
            weak_alpha = -0.025 * prev_5 + 0.004 * np.tanh(activity_state) - 0.006 * max(prev_20, 0)
            eps = rng.normal(0, 0.018)
            r = np.clip(0.0001 + beta * market_ret[t] + weak_alpha + eps, -0.105, 0.105)

            overnight = rng.normal(0, 0.004)
            open_price[t] = max(0.5, price * (1 + overnight))
            close[t] = max(0.5, open_price[t] * (1 + r))
            intraday_range = abs(rng.normal(0.018, 0.006))
            high[t] = max(open_price[t], close[t]) * (1 + intraday_range * rng.uniform(0.25, 1.00))
            low[t] = min(open_price[t], close[t]) * (1 - intraday_range * rng.uniform(0.25, 1.00))
            turnover[t] = np.clip(base_turnover * np.exp(activity_state * 0.25 + rng.normal(0, 0.20)), 0.0005, 0.25)
            volume[t] = float_shares * turnover[t]
            rets[t] = close[t] / price - 1
            price = close[t]

        amount = volume * close
        market_cap = close * float_shares
        for t, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open": float(open_price[t]),
                    "high": float(high[t]),
                    "low": float(low[t]),
                    "close": float(close[t]),
                    "volume": float(volume[t]),
                    "amount": float(amount[t]),
                    "turnover_rate": float(turnover[t]),
                    "market_cap": float(market_cap[t]),
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values(["date", "code"]).reset_index(drop=True)


def make_sample_data(config: dict) -> pd.DataFrame:
    """Create and save sample data according to the project config."""
    sample_cfg = SampleDataConfig(
        start_date=config["sample_data"]["start_date"],
        end_date=config["sample_data"]["end_date"],
        n_stocks=int(config["sample_data"]["n_stocks"]),
        seed=int(config["project"].get("seed", 42)),
    )
    df = generate_sample_daily_data(sample_cfg)
    save_parquet(df, config["paths"]["sample_data"])
    return df
