"""Run the complete AKShare-driven research pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Execute all project scripts in order."""
    steps = [
        "run_01_download_data.py",
        "run_02_build_features.py",
        "run_03_factor_analysis.py",
        "run_04_train_ml_models.py",
        "run_05_train_mlp.py",
        "run_06_backtest.py",
    ]
    for step in steps:
        print(f"\n===== Running {step} =====")
        subprocess.run([sys.executable, str(ROOT / "scripts" / step)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
