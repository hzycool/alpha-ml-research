"""Prediction metrics for cross-sectional return forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score


def prediction_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute point forecast and cross-sectional ranking metrics."""
    valid = y_true.notna() & y_pred.notna()
    y = y_true.loc[valid]
    p = y_pred.loc[valid]
    if len(y) < 2:
        return {"mse": np.nan, "r2": np.nan, "pearson_ic": np.nan, "spearman_rankic": np.nan}
    pearson = stats.pearsonr(y, p).statistic if y.nunique() > 1 and p.nunique() > 1 else np.nan
    spearman = stats.spearmanr(y, p).statistic if y.nunique() > 1 and p.nunique() > 1 else np.nan
    return {
        "mse": float(mean_squared_error(y, p)),
        "r2": float(r2_score(y, p)),
        "pearson_ic": float(pearson),
        "spearman_rankic": float(spearman),
    }


def daily_prediction_ic(df: pd.DataFrame, label_col: str, score_col: str) -> pd.DataFrame:
    """Compute daily Pearson IC and RankIC for model predictions."""
    rows: list[dict[str, object]] = []
    for date, g in df.groupby("date"):
        metrics = prediction_metrics(g[label_col], g[score_col])
        rows.append({"date": date, "ic": metrics["pearson_ic"], "rank_ic": metrics["spearman_rankic"]})
    return pd.DataFrame(rows)


def evaluate_predictions(df: pd.DataFrame, label_col: str, score_cols: list[str]) -> pd.DataFrame:
    """Evaluate multiple prediction columns."""
    rows: list[dict[str, object]] = []
    for score_col in score_cols:
        metrics = prediction_metrics(df[label_col], df[score_col])
        daily_ic = daily_prediction_ic(df, label_col, score_col)
        rows.append(
            {
                "model": score_col.replace("score_", ""),
                **metrics,
                "daily_ic_mean": daily_ic["ic"].mean(),
                "daily_rankic_mean": daily_ic["rank_ic"].mean(),
                "daily_rankic_std": daily_ic["rank_ic"].std(ddof=1),
            }
        )
    return pd.DataFrame(rows)
