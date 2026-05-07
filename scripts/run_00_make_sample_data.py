"""Generate sample A-share-like daily data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.make_sample_data import make_sample_data
from src.utils.io import load_config
from src.utils.logger import get_logger


def main() -> None:
    """Create sample data under data/sample/."""
    logger = get_logger("run_00")
    config = load_config()
    df = make_sample_data(config)
    logger.info("Sample data generated: %s rows, %s stocks", len(df), df["code"].nunique())


if __name__ == "__main__":
    main()
