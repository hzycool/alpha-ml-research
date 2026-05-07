"""Information coefficient analysis for cross-sectional alpha factors."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.io import resolve_path, save_csv


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    """Compute a robust correlation and return NaN when the cross-section is invalid."""
    valid = x.notna() & y.notna()
    if valid.sum() < 10 or x.loc[valid].nunique() <= 1 or y.loc[valid].nunique() <= 1:
        return np.nan
    if method == "pearson":
        return float(stats.pearsonr(x.loc[valid], y.loc[valid]).statistic)
    if method == "spearman":
        return float(stats.spearmanr(x.loc[valid], y.loc[valid]).statistic)
    raise ValueError(f"Unknown correlation method: {method}")


def compute_ic_timeseries(df: pd.DataFrame, factor_cols: list[str], label_col: str) -> pd.DataFrame:
    """Compute daily IC and RankIC for each factor."""
    records: list[dict[str, object]] = []
    for date, g in df.groupby("date"):
        y = g[label_col]
        for factor in factor_cols:
            records.append(
                {
                    "date": date,
                    "factor": factor,
                    "ic": _safe_corr(g[factor], y, "pearson"),
                    "rank_ic": _safe_corr(g[factor], y, "spearman"),
                }
            )
    return pd.DataFrame(records)


def summarize_ic(ic_ts: pd.DataFrame) -> pd.DataFrame:
    """Summarize IC and RankIC time series by factor."""
    rows: list[dict[str, object]] = []
    for factor, g in ic_ts.groupby("factor"):
        ic = g["ic"].dropna()
        rank_ic = g["rank_ic"].dropna()
        rows.append(
            {
                "factor": factor,
                "ic_mean": ic.mean(),
                "ic_std": ic.std(ddof=1),
                "icir": ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) != 0 else np.nan,
                "ic_t_stat": ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))) if len(ic) > 1 and ic.std(ddof=1) != 0 else np.nan,
                "ic_win_rate": (ic > 0).mean() if len(ic) else np.nan,
                "rankic_mean": rank_ic.mean(),
                "rankic_std": rank_ic.std(ddof=1),
                "rankicir": rank_ic.mean() / rank_ic.std(ddof=1) if rank_ic.std(ddof=1) != 0 else np.nan,
                "rankic_win_rate": (rank_ic > 0).mean() if len(rank_ic) else np.nan,
                "n_obs": len(rank_ic),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("rankic_mean", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def factor_correlation_matrix(df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """Compute the full-sample factor correlation matrix after preprocessing."""
    return df[factor_cols].corr(method="spearman")


def plot_ic_bars(summary: pd.DataFrame, figures_dir: str | Path) -> None:
    """Plot IC and RankIC bar charts."""
    figures_dir = resolve_path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_df = summary.sort_values("ic_mean")
    fig, ax = plt.subplots(figsize=(8, max(6, len(plot_df) * 0.18)))
    ax.barh(plot_df["factor"], plot_df["ic_mean"], color="#4169a8")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Single-factor Pearson IC Mean")
    ax.set_xlabel("IC Mean")
    fig.tight_layout()
    fig.savefig(figures_dir / "factor_ic_bar.png", dpi=180)
    plt.close(fig)

    plot_df = summary.sort_values("rankic_mean")
    fig, ax = plt.subplots(figsize=(8, max(6, len(plot_df) * 0.18)))
    ax.barh(plot_df["factor"], plot_df["rankic_mean"], color="#2f8f83")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Single-factor Spearman RankIC Mean")
    ax.set_xlabel("RankIC Mean")
    fig.tight_layout()
    fig.savefig(figures_dir / "factor_rankic_bar.png", dpi=180)
    plt.close(fig)


def plot_factor_corr(corr: pd.DataFrame, figures_dir: str | Path) -> None:
    """Plot factor correlation heatmap."""
    figures_dir = resolve_path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr.fillna(0), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.index, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Spearman Correlation Matrix of Alpha Factors")
    fig.tight_layout()
    fig.savefig(figures_dir / "factor_correlation_heatmap.png", dpi=180)
    plt.close(fig)


def run_ic_report(df: pd.DataFrame, factor_cols: list[str], label_col: str, tables_dir: str | Path, figures_dir: str | Path) -> pd.DataFrame:
    """Run single-factor IC analysis and save tables and figures."""
    ic_ts = compute_ic_timeseries(df, factor_cols, label_col)
    summary = summarize_ic(ic_ts)
    corr = factor_correlation_matrix(df, factor_cols)
    save_csv(ic_ts, Path(tables_dir) / "factor_ic_timeseries.csv")
    save_csv(summary, Path(tables_dir) / "factor_ic_summary.csv")
    save_csv(corr.reset_index(names="factor"), Path(tables_dir) / "factor_correlation_matrix.csv")
    plot_ic_bars(summary, figures_dir)
    plot_factor_corr(corr, figures_dir)
    return summary
