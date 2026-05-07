"""Prediction helpers for saved models."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


def predict_joblib_model(model_path: str | Path, df: pd.DataFrame, factor_cols: list[str]) -> pd.Series:
    """Load a joblib model and return predictions."""
    model = joblib.load(model_path)
    return pd.Series(model.predict(df[factor_cols]), index=df.index)
