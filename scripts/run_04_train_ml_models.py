"""Train tree-based machine learning alpha models."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.factor_dictionary import get_factor_list
from src.models.train_tree_models import train_tree_models
from src.utils.io import load_config, read_parquet
from src.utils.logger import get_logger


def main() -> None:
    """Train Random Forest, XGBoost, and LightGBM."""
    logger = get_logger("run_04")
    config = load_config()
    df = read_parquet(config["paths"]["processed_data"])
    predictions, metrics, boundaries = train_tree_models(df, get_factor_list(), config["labels"]["primary"], config)
    logger.info("Tree model training complete: %s predictions", len(predictions))
    logger.info("Chronological split: %s", {k: str(v.date()) for k, v in boundaries.items()})
    logger.info("Prediction metrics:\n%s", metrics.to_string(index=False))


if __name__ == "__main__":
    main()
