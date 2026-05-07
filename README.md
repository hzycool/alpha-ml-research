# A-Share Machine Learning Alpha Research and Backtesting Framework

机器学习驱动的 A 股多因子 Alpha 挖掘与回测框架。

This repository implements an end-to-end **Machine Learning Alpha Research** workflow for public A-share daily price-volume data downloaded through AKShare. It is designed as a reproducible quant research project for QR internship interviews: data download, cleaning, stock-pool filtering, label construction, alpha factor engineering, IC/RankIC analysis, tree/MLP model training, chronological out-of-sample validation, portfolio backtesting, transaction-cost sensitivity, robustness checks, and a LaTeX research report.

> This project is a research framework, not investment advice. The included AKShare run uses public historical daily data and a medium-size demonstration universe; it does not claim stable profitability or production tradability.

## Project Overview

**English title:** A-Share Machine Learning Alpha Research and Backtesting Framework  
**Chinese title:** 机器学习驱动的 A 股多因子 Alpha 挖掘与回测框架

The default pipeline downloads a medium-size A-share universe through AKShare and evaluates whether machine learning models can extract weak cross-sectional alpha signals from daily price-volume factors under chronological validation and transaction-cost assumptions.

## Research Question

Can machine learning models extract robust alpha signals from A-share daily price-volume factors and improve cross-sectional return prediction under realistic out-of-sample validation and transaction cost assumptions?

中文表述：机器学习模型能否基于 A 股日频价量因子提取具有样本外稳定性的 Alpha 信号，并在考虑交易成本后改善横截面收益预测与组合回测表现？

## Data Source

Default data source:

- Provider: AKShare public A-share daily data
- Interface used: `ak.stock_zh_a_daily`, with qfq adjustment
- Period: 2021-01-04 to 2024-12-31
- Candidate universe: 100 liquid A-share names from a built-in demo universe
- Valid downloaded stocks in this run: 98
- Rows downloaded: 94,382
- Saved file: `data/sample/a_share_akshare_daily.parquet`

The AKShare spot-universe endpoint was unstable in this environment, so the project uses a fixed medium-size demo universe and downloads historical data stock by stock. Two names failed due to AKShare decode issues and were excluded. This is disclosed rather than silently imputed.

## Why Machine Learning Alpha Research

Daily stock-return prediction is noisy and low signal. This project treats machine learning as a research tool for cross-sectional ranking, not as a black-box profit machine.

Model roles:

- Random Forest: bagging-based nonlinear benchmark.
- XGBoost: classic boosting tree model.
- LightGBM: efficient GBDT model for tabular factor panels.
- PyTorch MLP: simple deep learning baseline.

We do not start with Transformers because daily tabular alpha data is noisy, relatively small, and cross-sectional. GBDT models are more natural first baselines for this setting, while sequence models need stronger data and stricter validation.

## Repository Structure

```text
alpha-ml-research/
├── README.md
├── requirements.txt
├── config.yaml
├── .gitignore
├── LICENSE
├── data/
│   ├── sample/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── analysis/
│   ├── models/
│   ├── backtest/
│   └── utils/
├── scripts/
└── reports/
    ├── figures/
    ├── tables/
    ├── final_report.tex
    ├── references.bib
    └── final_report.pdf
```

`resume/` and `GITHUB_UPLOAD.md` are intentionally ignored for GitHub upload per the project owner's request.

## Alpha Factors

The framework constructs 36 daily price-volume alpha factors, including:

- Momentum/reversal: `ret_1`, `ret_3`, `ret_5`, `ret_10`, `ret_20`, `ret_60`, `ret_120`
- Volatility/range: rolling volatility and high-low amplitude
- Liquidity/activity: rolling volume, amount, and turnover means
- Price-volume relation: rolling return-volume and return-amount correlations
- Activity change: short-term vs medium-term volume/amount/turnover change
- Price location: close position relative to recent high-low range
- Risk-adjusted return: return divided by realized volatility

See `reports/tables/factor_dictionary.csv` for formulas and economic interpretations.

## Factor Preprocessing

For each trading date, every factor is winsorized at the 1% and 99% cross-sectional quantiles and then z-scored. The code includes a market-cap neutralization interface, but this AKShare demo does not use it because qfq-adjusted prices combined with outstanding shares do not constitute a clean point-in-time market-cap risk model.

## Avoiding Look-ahead Bias

- Factors use only date `t` and earlier data.
- Labels start from `t+1`: `y_5d = Open_{t+6} / Open_{t+1} - 1`.
- Rolling calculations are grouped by stock code.
- Winsorization and z-score are date-wise cross-sectional operations.
- Model training uses chronological train/validation/test splits.
- No random `train_test_split` is used.
- The test set is only used for final evaluation.
- Financial statement data without announcement dates is excluded.
- A fixed demo universe may introduce universe/survivorship bias, which is disclosed as a limitation.

## Models

| Model | Role |
|---|---|
| Random Forest | Bagging-based nonlinear benchmark |
| XGBoost | Boosting tree model for nonlinear interactions |
| LightGBM | Efficient GBDT model for tabular panel data |
| PyTorch MLP | Simple deep learning baseline |

Financial prediction should not be judged only by MSE or R². Low or negative out-of-sample R² is common. The project focuses on daily IC/RankIC, grouping behavior, transaction-cost-adjusted performance, turnover, drawdown, and robustness.

## Backtesting Rules

- Rebalance every 5 trading days.
- On date `t`, use model scores generated after close.
- Select the top 10% stocks by predicted score.
- Enter from `t+1` open and hold for 5 trading days.
- Equal-weight selected stocks.
- Benchmark is the equal-weight universe portfolio.
- Default single-side transaction cost is 10 bps.
- Cost sensitivity is tested at 5, 10, and 20 bps.

## Results

Chronological split:

- Train: 2021-07-05 to 2023-08-01
- Validation: 2023-08-02 to 2024-04-15
- Test: 2024-04-16 to 2024-12-23

Out-of-sample prediction metrics:

| Model | Daily IC | Daily RankIC | R² |
|---|---:|---:|---:|
| Random Forest | 0.0385 | 0.0258 | -0.0042 |
| XGBoost | 0.0305 | 0.0208 | -0.0082 |
| LightGBM | 0.0252 | 0.0245 | -0.0162 |
| MLP | 0.0304 | 0.0430 | -0.0123 |

10 bps cost-adjusted backtest metrics:

| Model | Annual Return | Sharpe | Max Drawdown | Turnover |
|---|---:|---:|---:|---:|
| Random Forest | 0.2978 | 0.9111 | -0.1754 | 1.0353 |
| XGBoost | 0.2381 | 0.8264 | -0.1574 | 1.0647 |
| LightGBM | 0.1523 | 0.5947 | -0.1872 | 1.0941 |
| MLP | 0.2956 | 1.5133 | -0.0846 | 1.1882 |

The equal-weight benchmark annual return over the same test window is 0.3176. In this AKShare run, the ML long-only portfolios do not robustly outperform the benchmark after costs. This is an important research finding: weak positive RankIC does not automatically translate into benchmark-beating tradable alpha.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full AKShare pipeline:

```bash
python scripts/run_all.py
```

Or run step by step:

```bash
python scripts/run_01_download_data.py
python scripts/run_02_build_features.py
python scripts/run_03_factor_analysis.py
python scripts/run_04_train_ml_models.py
python scripts/run_05_train_mlp.py
python scripts/run_06_backtest.py
```

Optional offline sample-data generation:

```bash
python scripts/run_00_make_sample_data.py
```

Compile the report:

```bash
cd reports
xelatex final_report.tex
bibtex final_report
xelatex final_report.tex
xelatex final_report.tex
```

## Limitations

- AKShare is a public data interface and may be unstable or change fields.
- The demo universe is fixed and may have universe/survivorship bias.
- The market-cap neutralization interface is retained, but strict point-in-time risk neutralization is not used in this run.
- Transaction costs are simplified and do not include market impact.
- Limit-up/limit-down, suspension, ST status, and realistic execution constraints are not fully modeled.
- No real-money or live trading validation is performed.

## Future Work

- Use a larger point-in-time A-share universe.
- Add industry and style risk neutralization.
- Add limit status, suspension, ST, and listing-age filters.
- Implement walk-forward retraining.
- Add turnover-aware portfolio optimization.
- Test intraday or high-frequency order-book features.
- Compare GBDT models with Temporal CNN or Transformer only after stronger data validation.

## GitHub Upload

This repository is prepared so that `resume/` and `GITHUB_UPLOAD.md` are ignored. Large raw data, processed matrices, and model artifacts are also ignored.

```bash
git init
git add .
git commit -m "Initial commit: AKShare machine learning alpha research framework"
gh auth login
gh repo create alpha-ml-research --public --source=. --remote=origin --push
```

## Disclaimer

This repository is for research and interview demonstration purposes. It is not investment advice and does not claim stable real-market profitability.
