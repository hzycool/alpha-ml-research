"""Run model-score portfolio backtests and robustness checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.cost import run_cost_sensitivity
from src.backtest.portfolio import run_long_only_backtest, run_model_backtests, summarize_backtest
from src.models.evaluate import evaluate_predictions
from src.utils.io import load_config, read_parquet, resolve_path, save_csv
from src.utils.logger import get_logger


def _merge_all_labels(predictions: pd.DataFrame, features: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Ensure prediction table contains all forward-return labels."""
    label_cols = [f"y_{h}d" for h in horizons]
    labels = features[["date", "code", *label_cols]].drop_duplicates(["date", "code"])
    base_cols = ["date", "code"]
    overlap = [col for col in label_cols if col in predictions.columns]
    merged = predictions.drop(columns=overlap, errors="ignore").merge(labels, on=base_cols, how="left")
    return merged


def _write_latex_fragments(tables_dir: str) -> None:
    """Write compact table fragments used by the LaTeX report."""
    tables_path = resolve_path(tables_dir)
    for csv_name, tex_name, cols in [
        (
            "model_comparison_metrics.csv",
            "model_comparison_metrics.tex",
            ["model", "annual_return", "annual_volatility", "sharpe", "max_drawdown", "turnover"],
        ),
        (
            "prediction_metrics_all.csv",
            "prediction_metrics_all.tex",
            ["model", "mse", "r2", "daily_ic_mean", "daily_rankic_mean"],
        ),
        (
            "cost_sensitivity.csv",
            "cost_sensitivity.tex",
            ["cost_bps", "annual_return", "sharpe", "max_drawdown", "turnover"],
        ),
        (
            "holding_period_sensitivity.csv",
            "holding_period_sensitivity.tex",
            ["horizon", "annual_return", "sharpe", "max_drawdown", "turnover"],
        ),
    ]:
        csv_path = tables_path / csv_name
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        use_cols = [c for c in cols if c in df.columns]
        tex_df = df[use_cols].copy()
        if "model" in tex_df.columns:
            tex_df["model"] = tex_df["model"].replace(
                {
                    "random_forest": "Random Forest",
                    "xgboost": "XGBoost",
                    "lightgbm": "LightGBM",
                    "mlp": "MLP",
                }
            )
        for col in tex_df.select_dtypes(include="number").columns:
            tex_df[col] = tex_df[col].map(lambda x: f"{x:.4f}")
        pretty_cols = {
            "model": "Model",
            "mse": "MSE",
            "r2": "$R^2$",
            "daily_ic_mean": "Daily IC",
            "daily_rankic_mean": "Daily RankIC",
            "annual_return": "Annual Return",
            "annual_volatility": "Annual Vol",
            "sharpe": "Sharpe",
            "max_drawdown": "Max DD",
            "turnover": "Turnover",
            "cost_bps": "Cost bps",
            "horizon": "Horizon",
        }
        tex_df = tex_df.rename(columns={c: pretty_cols.get(c, c) for c in tex_df.columns})
        fragment = tex_df.to_latex(index=False, escape=False)
        (tables_path / tex_name).write_text(fragment, encoding="utf-8")


def main() -> None:
    """Run backtests, cost sensitivity, and holding-period robustness."""
    logger = get_logger("run_06")
    config = load_config()
    predictions = read_parquet(config["paths"]["predictions"])
    features = read_parquet(config["paths"]["processed_data"])
    predictions = _merge_all_labels(predictions, features, list(config["labels"]["horizons"]))
    score_cols = [col for col in predictions.columns if col.startswith("score_")]
    if not score_cols:
        raise RuntimeError("No model score columns found. Run model training scripts first.")

    pred_metrics = evaluate_predictions(predictions.dropna(subset=[config["labels"]["primary"]]), config["labels"]["primary"], score_cols)
    save_csv(pred_metrics, Path(config["paths"]["tables_dir"]) / "prediction_metrics_all.csv")

    metrics, _ = run_model_backtests(predictions, score_cols, config, label_col=config["labels"]["primary"])
    logger.info("Backtest metrics:\n%s", metrics.to_string(index=False))

    preferred = "score_lightgbm" if "score_lightgbm" in score_cols else score_cols[0]
    run_cost_sensitivity(predictions, preferred, config["labels"]["primary"], config)

    rows = []
    periods_per_year_base = float(config["backtest"]["annual_trading_days"]) / int(config["backtest"]["rebalance_every"])
    for horizon in config["labels"]["horizons"]:
        label_col = f"y_{horizon}d"
        if label_col not in predictions.columns:
            continue
        bt = run_long_only_backtest(
            predictions,
            score_col=preferred,
            label_col=label_col,
            rebalance_every=max(1, int(horizon)),
            top_quantile=float(config["backtest"]["top_quantile"]),
            cost=float(config["backtest"]["default_cost"]),
        )
        summary = summarize_backtest(bt, preferred.replace("score_", ""), periods_per_year_base if horizon == 5 else 252 / max(1, int(horizon)))
        summary["horizon"] = f"{horizon}d"
        rows.append(summary)
    if rows:
        save_csv(pd.DataFrame(rows), Path(config["paths"]["tables_dir"]) / "holding_period_sensitivity.csv")

    _write_latex_fragments(config["paths"]["tables_dir"])
    logger.info("Backtest and robustness checks complete. Preferred model for sensitivity: %s", preferred)


if __name__ == "__main__":
    main()
