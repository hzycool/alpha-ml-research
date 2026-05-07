"""Input-output helpers for reproducible project scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    path = resolve_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load the YAML project configuration."""
    config_path = resolve_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a DataFrame as parquet and create parent directories."""
    out_path = resolve_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a parquet file relative to the repository root."""
    in_path = resolve_path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {in_path}")
    return pd.read_parquet(in_path)


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Save a DataFrame as CSV with UTF-8 encoding."""
    out_path = resolve_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=index, encoding="utf-8-sig")
    return out_path


def save_latex_table(df: pd.DataFrame, path: str | Path, caption: str | None = None) -> Path:
    """Save a small DataFrame as a LaTeX table fragment."""
    out_path = resolve_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tex = df.to_latex(index=False, escape=False, caption=caption)
    out_path.write_text(tex, encoding="utf-8")
    return out_path
