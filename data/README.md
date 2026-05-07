# Data Directory

This project supports two data modes.

- `data/sample/a_share_akshare_daily.parquet`: small public AKShare daily panel used by the default demo pipeline. It is intentionally medium-size for GitHub reproducibility.
- `data/sample/a_share_sample_daily.parquet`: optional synthetic A-share-like panel that can be regenerated for offline pipeline checks.
- `data/raw/`: optional raw public data downloaded through AKShare. Large raw files are ignored by Git.
- `data/processed/`: intermediate factor matrices, predictions, and model-ready tables. These files are generated locally and ignored by Git.

The included AKShare demo panel is public historical data, but the results are still research demonstrations and must not be interpreted as investment advice or stable tradable evidence.
