"""Long-only portfolio backtesting using model scores."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest.metrics import performance_metrics
from src.utils.io import resolve_path, save_csv


def _weights_for_selection(codes: list[str]) -> dict[str, float]:
    """Create equal weights for selected codes."""
    if not codes:
        return {}
    weight = 1.0 / len(codes)
    return {code: weight for code in codes}


def _portfolio_turnover(prev: dict[str, float], new: dict[str, float]) -> float:
    """Compute two-sided portfolio weight turnover."""
    all_codes = set(prev) | set(new)
    return float(sum(abs(new.get(code, 0.0) - prev.get(code, 0.0)) for code in all_codes))


def run_long_only_backtest(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "y_5d",
    rebalance_every: int = 5,
    top_quantile: float = 0.10,
    cost: float = 0.001,
) -> pd.DataFrame:
    """Run 5-day rebalanced top-quantile long-only backtest.

    On date t, scores are observed after close; y_5d measures the return from
    t+1 open to t+6 open, matching the no-look-ahead portfolio rule.
    """
    needed = ["date", "code", label_col, score_col]
    data = df.dropna(subset=needed).copy()
    data["date"] = pd.to_datetime(data["date"])
    rebalance_dates = sorted(data["date"].unique())[::rebalance_every]
    prev_weights: dict[str, float] = {}
    rows: list[dict[str, object]] = []

    for date in rebalance_dates:
        g = data[data["date"] == date].copy()
        if len(g) < 10:
            continue
        n_select = max(1, int(np.ceil(len(g) * top_quantile)))
        selected = g.sort_values(score_col, ascending=False).head(n_select)
        weights = _weights_for_selection(selected["code"].tolist())
        turnover = _portfolio_turnover(prev_weights, weights)
        gross_ret = float(selected[label_col].mean())
        benchmark_ret = float(g[label_col].mean())
        cost_paid = turnover * cost
        net_ret = gross_ret - cost_paid
        rows.append(
            {
                "date": date,
                "gross_return": gross_ret,
                "cost": cost_paid,
                "net_return": net_ret,
                "benchmark_return": benchmark_ret,
                "turnover": turnover,
                "n_selected": n_select,
            }
        )
        prev_weights = weights

    result = pd.DataFrame(rows).sort_values("date")
    result["strategy_nav"] = (1 + result["net_return"]).cumprod()
    result["gross_nav"] = (1 + result["gross_return"]).cumprod()
    result["benchmark_nav"] = (1 + result["benchmark_return"]).cumprod()
    result["excess_nav"] = result["strategy_nav"] / result["benchmark_nav"]
    result["drawdown"] = result["strategy_nav"] / result["strategy_nav"].cummax() - 1
    result["score_col"] = score_col
    return result


def summarize_backtest(bt: pd.DataFrame, model_name: str, periods_per_year: float) -> dict[str, float | str]:
    """Summarize backtest performance for one model."""
    strategy = performance_metrics(bt["net_return"], bt["strategy_nav"], bt["turnover"], periods_per_year)
    benchmark = performance_metrics(bt["benchmark_return"], bt["benchmark_nav"], None, periods_per_year)
    return {
        "model": model_name,
        "total_return": strategy["total_return"],
        "annual_return": strategy["annual_return"],
        "annual_volatility": strategy["annual_volatility"],
        "sharpe": strategy["sharpe"],
        "max_drawdown": strategy["max_drawdown"],
        "turnover": strategy["turnover"],
        "benchmark_annual_return": benchmark["annual_return"],
        "excess_total_return": float(bt["excess_nav"].iloc[-1] - 1) if len(bt) else np.nan,
    }


def plot_backtest(bt: pd.DataFrame, figures_dir: str, prefix: str = "") -> None:
    """Save NAV, excess NAV, and drawdown plots."""
    figures_dir = resolve_path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bt["date"], bt["strategy_nav"], label="Strategy NAV", color="#2f6f9f")
    ax.plot(bt["date"], bt["benchmark_nav"], label="Benchmark NAV", color="#777777")
    ax.set_title("Strategy vs Benchmark NAV")
    ax.set_ylabel("NAV")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}strategy_nav.png", dpi=180)
    fig.savefig(figures_dir / f"{prefix}benchmark_nav.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bt["date"], bt["excess_nav"], color="#5b8c5a")
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_title("Excess NAV")
    ax.set_ylabel("Strategy / Benchmark")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}excess_nav.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(bt["date"], bt["drawdown"], 0, color="#a34d4d", alpha=0.55)
    ax.set_title("Strategy Drawdown")
    ax.set_ylabel("Drawdown")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}drawdown.png", dpi=180)
    plt.close(fig)


def plot_model_comparison(backtests: dict[str, pd.DataFrame], figures_dir: str) -> None:
    """Plot model comparison NAV curves."""
    figures_dir = resolve_path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, bt in backtests.items():
        ax.plot(bt["date"], bt["strategy_nav"], label=name, linewidth=1.2)
    ax.set_title("Model Comparison NAV")
    ax.set_ylabel("NAV")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures_dir / "model_comparison_nav.png", dpi=180)
    plt.close(fig)


def run_model_backtests(
    predictions: pd.DataFrame,
    score_cols: list[str],
    config: dict,
    label_col: str = "y_5d",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Backtest all available model score columns."""
    bt_cfg = config["backtest"]
    periods_per_year = float(bt_cfg["annual_trading_days"]) / int(bt_cfg["rebalance_every"])
    rows: list[dict[str, float | str]] = []
    backtests: dict[str, pd.DataFrame] = {}

    for score_col in score_cols:
        if score_col not in predictions.columns:
            continue
        model_name = score_col.replace("score_", "")
        bt = run_long_only_backtest(
            predictions,
            score_col=score_col,
            label_col=label_col,
            rebalance_every=int(bt_cfg["rebalance_every"]),
            top_quantile=float(bt_cfg["top_quantile"]),
            cost=float(bt_cfg["default_cost"]),
        )
        if bt.empty:
            continue
        backtests[model_name] = bt
        rows.append(summarize_backtest(bt, model_name, periods_per_year))
        save_csv(bt, f"reports/tables/backtest_{model_name}.csv")

    metrics = pd.DataFrame(rows)
    save_csv(metrics, "reports/tables/backtest_metrics.csv")
    save_csv(metrics, "reports/tables/model_comparison_metrics.csv")
    if backtests:
        first_name = next(iter(backtests))
        plot_backtest(backtests[first_name], config["paths"]["figures_dir"])
        plot_model_comparison(backtests, config["paths"]["figures_dir"])
    return metrics, backtests
