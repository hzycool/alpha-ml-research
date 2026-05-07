"""Orchestrator for single-factor analysis reports."""

from __future__ import annotations

import pandas as pd

from src.analysis.group_analysis import run_group_report
from src.analysis.ic_analysis import run_ic_report


def run_factor_analysis(
    df: pd.DataFrame,
    factor_cols: list[str],
    label_col: str,
    tables_dir: str,
    figures_dir: str,
    n_groups: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run IC analysis and group backtest for the strongest RankIC factor."""
    summary = run_ic_report(df, factor_cols, label_col, tables_dir, figures_dir)
    top_factor = summary.iloc[0]["factor"]
    groups = run_group_report(df, top_factor, label_col, tables_dir, figures_dir, n_groups=n_groups)
    return summary, groups
