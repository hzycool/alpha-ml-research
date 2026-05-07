"""Run single-factor IC and grouping analysis."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.factor_report import run_factor_analysis
from src.features.factor_dictionary import get_factor_list
from src.utils.io import load_config, read_parquet
from src.utils.logger import get_logger


def main() -> None:
    """Generate factor analysis tables and figures."""
    logger = get_logger("run_03")
    config = load_config()
    df = read_parquet(config["paths"]["processed_data"])
    label_col = config["labels"]["primary"]
    summary, _ = run_factor_analysis(
        df,
        get_factor_list(),
        label_col,
        config["paths"]["tables_dir"],
        config["paths"]["figures_dir"],
        n_groups=int(config["factor_analysis"]["n_groups"]),
    )
    logger.info("Factor analysis complete. Top RankIC factor: %s", summary.iloc[0]["factor"])


if __name__ == "__main__":
    main()
