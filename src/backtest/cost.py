"""Transaction cost sensitivity analysis."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.portfolio import run_long_only_backtest, summarize_backtest
from src.utils.io import resolve_path, save_csv


def run_cost_sensitivity(predictions: pd.DataFrame, score_col: str, label_col: str, config: dict) -> pd.DataFrame:
    """Evaluate strategy performance under multiple single-side cost assumptions."""
    bt_cfg = config["backtest"]
    periods_per_year = float(bt_cfg["annual_trading_days"]) / int(bt_cfg["rebalance_every"])
    rows: list[dict[str, float | str]] = []
    for cost in bt_cfg["cost_levels"]:
        bt = run_long_only_backtest(
            predictions,
            score_col=score_col,
            label_col=label_col,
            rebalance_every=int(bt_cfg["rebalance_every"]),
            top_quantile=float(bt_cfg["top_quantile"]),
            cost=float(cost),
        )
        summary = summarize_backtest(bt, score_col.replace("score_", ""), periods_per_year)
        summary["cost_bps"] = float(cost) * 10000
        rows.append(summary)
    result = pd.DataFrame(rows)
    save_csv(result, "reports/tables/cost_sensitivity.csv")

    fig_dir = resolve_path(config["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result["cost_bps"], result["annual_return"], marker="o", label="Annual return")
    ax.plot(result["cost_bps"], result["sharpe"], marker="s", label="Sharpe")
    ax.set_title("Transaction Cost Sensitivity")
    ax.set_xlabel("Single-side cost (bps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "cost_sensitivity.png", dpi=180)
    plt.close(fig)
    return result
