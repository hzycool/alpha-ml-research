"""Model factory for tree-based machine learning alpha models."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor


def get_tree_model(name: str, params: dict):
    """Instantiate a supported tree model."""
    model_name = name.lower()
    if model_name in {"random_forest", "rf"}:
        return RandomForestRegressor(**params)
    if model_name in {"xgboost", "xgb"}:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is not installed. Install with `pip install xgboost`.") from exc
        return XGBRegressor(**params)
    if model_name in {"lightgbm", "lgbm"}:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("lightgbm is not installed. Install with `pip install lightgbm`.") from exc
        return LGBMRegressor(**params)
    raise ValueError(f"Unsupported model: {name}")
