"""Train PyTorch MLP baseline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.factor_dictionary import get_factor_list
from src.models.train_mlp import train_mlp_model
from src.utils.io import load_config, read_parquet
from src.utils.logger import get_logger


def main() -> None:
    """Train and evaluate the MLP baseline."""
    logger = get_logger("run_05")
    config = load_config()
    df = read_parquet(config["paths"]["processed_data"])
    _, metrics = train_mlp_model(df, get_factor_list(), config["labels"]["primary"], config)
    logger.info("MLP training complete:\n%s", metrics.to_string(index=False))


if __name__ == "__main__":
    main()
