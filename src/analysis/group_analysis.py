"""Single-factor and model-score grouping analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import resolve_path, save_csv


def assign_quantile_groups(scores: pd.Series, n_groups: int) -> pd.Series:
    """Assign 1..n quantile groups for one cross-section."""
    valid = scores.dropna()
    out = pd.Series(np.nan, index=scores.index)
    if valid.nunique() < n_groups or len(valid) < n_groups * 2:
        return out
    try:
        out.loc[valid.index] = pd.qcut(valid.rank(method="first"), n_groups, labels=False) + 1
    except ValueError:
        return out
    return out


def compute_group_returns(df: pd.DataFrame, score_col: str, label_col: str, n_groups: int = 5) -> pd.DataFrame:
    """Compute mean forward returns by daily score quantile group."""
    frames: list[pd.DataFrame] = []
    for date, g in df.groupby("date"):
        group = assign_quantile_groups(g[score_col], n_groups)
        tmp = g[["date", label_col]].copy()
        tmp["group"] = group
        tmp = tmp.dropna(subset=["group", label_col])
        if tmp.empty:
            continue
        frames.append(tmp.groupby(["date", "group"], as_index=False)[label_col].mean())
    if not frames:
        raise ValueError(f"No valid quantile groups for {score_col}")
    out = pd.concat(frames, ignore_index=True)
    out["group"] = out["group"].astype(int)
    return out.rename(columns={label_col: "forward_return"})


def plot_group_results(group_returns: pd.DataFrame, figures_dir: str | Path, prefix: str = "") -> None:
    """Plot average quantile returns and group NAV curves."""
    figures_dir = resolve_path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    avg = group_returns.groupby("group")["forward_return"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(avg["group"].astype(str), avg["forward_return"], color="#5b8c5a")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Average Forward Return by Quantile Group")
    ax.set_xlabel("Group (low to high score)")
    ax.set_ylabel("Mean forward return")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}group_return_bar.png", dpi=180)
    plt.close(fig)

    pivot = group_returns.pivot(index="date", columns="group", values="forward_return").sort_index()
    nav = (1 + pivot.fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(7, 4))
    for col in nav.columns:
        ax.plot(nav.index, nav[col], label=f"G{col}", linewidth=1.2)
    ax.set_title("Quantile Group NAV Curves")
    ax.set_ylabel("NAV")
    ax.legend(ncol=3, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}group_nav_curve.png", dpi=180)
    plt.close(fig)

    high = pivot.max(axis=1)
    low = pivot.min(axis=1)
    ls = (1 + (high - low).fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ls.index, ls, color="#a34d4d")
    ax.set_title("Long-Short Quantile NAV")
    ax.set_ylabel("NAV")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}long_short_nav.png", dpi=180)
    plt.close(fig)


def run_group_report(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    tables_dir: str | Path,
    figures_dir: str | Path,
    n_groups: int = 5,
    prefix: str = "",
) -> pd.DataFrame:
    """Run grouping analysis and save outputs."""
    group_returns = compute_group_returns(df, score_col, label_col, n_groups)
    save_csv(group_returns, Path(tables_dir) / f"{prefix}group_returns.csv")
    plot_group_results(group_returns, figures_dir, prefix=prefix)
    return group_returns
