"""Training utilities for Random Forest, XGBoost, and LightGBM."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.features.preprocessing import drop_model_missing
from src.models.evaluate import evaluate_predictions
from src.models.model_zoo import get_tree_model
from src.utils.io import resolve_path, save_csv, save_parquet


def time_series_split_dates(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> dict[str, pd.Timestamp]:
    """Create chronological train/validation/test date boundaries."""
    dates = pd.Series(pd.to_datetime(df["date"].unique())).sort_values().reset_index(drop=True)
    if len(dates) < 20:
        raise ValueError("Not enough dates for chronological split.")
    train_end_idx = int(len(dates) * train_ratio) - 1
    val_end_idx = int(len(dates) * (train_ratio + val_ratio)) - 1
    return {
        "train_start": dates.iloc[0],
        "train_end": dates.iloc[train_end_idx],
        "val_start": dates.iloc[train_end_idx + 1],
        "val_end": dates.iloc[val_end_idx],
        "test_start": dates.iloc[val_end_idx + 1],
        "test_end": dates.iloc[-1],
    }


def split_by_time(df: pd.DataFrame, boundaries: dict[str, pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets by date."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    train = out[(out["date"] >= boundaries["train_start"]) & (out["date"] <= boundaries["train_end"])]
    val = out[(out["date"] >= boundaries["val_start"]) & (out["date"] <= boundaries["val_end"])]
    test = out[(out["date"] >= boundaries["test_start"]) & (out["date"] <= boundaries["test_end"])]
    return train, val, test


def train_tree_models(
    df: pd.DataFrame,
    factor_cols: list[str],
    label_col: str,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Timestamp]]:
    """Train tree models using chronological validation and return test predictions."""
    model_df = drop_model_missing(df, factor_cols, label_col)
    boundaries = time_series_split_dates(
        model_df,
        float(config["validation"]["train_ratio"]),
        float(config["validation"]["val_ratio"]),
    )
    train, val, test = split_by_time(model_df, boundaries)
    model_dir = resolve_path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    predictions = test[["date", "code", label_col]].copy()
    metrics_rows: list[pd.DataFrame] = []

    model_specs = {
        "random_forest": config["models"]["random_forest"],
        "xgboost": config["models"]["xgboost"],
        "lightgbm": config["models"]["lightgbm"],
    }
    for name, params in model_specs.items():
        try:
            model = get_tree_model(name, params)
            fit_kwargs = {}
            if name == "lightgbm":
                fit_kwargs["eval_set"] = [(val[factor_cols], val[label_col])]
                fit_kwargs["eval_metric"] = "l2"
            elif name == "xgboost":
                fit_kwargs["eval_set"] = [(val[factor_cols], val[label_col])]
                fit_kwargs["verbose"] = False
            model.fit(train[factor_cols], train[label_col], **fit_kwargs)
            score_col = f"score_{name}"
            predictions[score_col] = model.predict(test[factor_cols])
            joblib.dump(model, model_dir / f"{name}.joblib")
            metrics = evaluate_predictions(predictions.dropna(subset=[score_col]), label_col, [score_col])
            metrics_rows.append(metrics)
        except Exception as exc:  # noqa: BLE001
            metrics_rows.append(pd.DataFrame([{"model": name, "error": str(exc)}]))

    metrics_df = pd.concat(metrics_rows, ignore_index=True)
    save_parquet(predictions, config["paths"]["predictions"])
    save_csv(metrics_df, Path(config["paths"]["tables_dir"]) / "prediction_metrics_tree.csv")

    split_df = pd.DataFrame([{k: v.date().isoformat() for k, v in boundaries.items()}])
    save_csv(split_df, Path(config["paths"]["tables_dir"]) / "time_split.csv")
    return predictions, metrics_df, boundaries
