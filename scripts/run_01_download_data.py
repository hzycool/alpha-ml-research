"""Download the configured medium-size A-share universe using AKShare."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.download import FALLBACK_A_SHARE_UNIVERSE, download_akshare_daily, get_akshare_universe
from src.utils.io import load_config
from src.utils.logger import get_logger


def main() -> None:
    """Download the configured AKShare universe."""
    logger = get_logger("run_01")
    config = load_config()
    ak_cfg = config["akshare"]
    codes = (
        get_akshare_universe(int(ak_cfg["universe_size"]))
        if bool(ak_cfg.get("use_spot_universe", False))
        else FALLBACK_A_SHARE_UNIVERSE[: int(ak_cfg["universe_size"])]
    )
    logger.info("Downloading AKShare daily data for %s candidate stocks", len(codes))
    df = download_akshare_daily(
        codes=codes,
        start_date=str(ak_cfg["start_date"]),
        end_date=str(ak_cfg["end_date"]),
        out_path=config["paths"]["raw_data"],
        adjust=str(ak_cfg.get("adjust", "qfq")),
        source=str(ak_cfg.get("source", "sina")),
        retry_times=int(ak_cfg.get("retry_times", 3)),
        retry_sleep_seconds=float(ak_cfg.get("retry_sleep_seconds", 1.5)),
        min_valid_stocks=int(ak_cfg.get("min_valid_stocks", 80)),
    )
    logger.info("Downloaded AKShare data: %s rows, %s stocks", len(df), df["code"].nunique())


if __name__ == "__main__":
    main()
