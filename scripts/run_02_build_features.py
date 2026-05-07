"""Build labels, alpha factors, and cross-sectional preprocessed feature matrix."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import filter_stock_pool
from src.features.factor_builder import add_forward_returns, build_alpha_factors
from src.features.factor_dictionary import factor_dictionary_frame
from src.features.preprocessing import preprocess_factors
from src.utils.io import load_config, read_parquet, save_csv, save_parquet
from src.utils.logger import get_logger


def main() -> None:
    """Create the model-ready factor matrix."""
    logger = get_logger("run_02")
    config = load_config()
    data_mode = config["project"].get("data_mode", "sample")
    source_path = config["paths"]["sample_data"] if data_mode == "sample" else config["paths"]["raw_data"]
    df = read_parquet(source_path)
    logger.info("Loaded %s data: %s rows", data_mode, len(df))

    filtered = filter_stock_pool(
        df,
        min_history_days=int(config["data"]["min_history_days"]),
        min_price=float(config["data"]["min_price"]),
        max_missing_ratio=float(config["data"]["max_missing_ratio"]),
    )
    with_labels = add_forward_returns(
        filtered,
        horizons=list(config["labels"]["horizons"]),
        price_col_preference=config["labels"].get("price_col_preference", "open"),
    )
    features, factor_cols = build_alpha_factors(with_labels)
    processed = preprocess_factors(
        features,
        factor_cols,
        winsor_lower=float(config["features"]["winsorize_lower"]),
        winsor_upper=float(config["features"]["winsorize_upper"]),
        neutralize_market_cap=bool(config["features"]["neutralize_market_cap"]),
    )
    save_parquet(processed, config["paths"]["processed_data"])
    save_csv(factor_dictionary_frame(), Path(config["paths"]["tables_dir"]) / "factor_dictionary.csv")
    logger.info("Feature matrix saved: %s rows, %s factors", len(processed), len(factor_cols))


if __name__ == "__main__":
    main()
