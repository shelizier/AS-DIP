"""Aggregate AS-DIP output folders into one summary table and figure."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl_cache"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.reporting import collect_experiment_rows, export_aggregate_table, plot_aggregate_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate AS-DIP experiment results.")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--save-dir", default="outputs/aggregate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_experiment_rows(args.outputs_dir)
    save_dir = Path(args.save_dir)
    export_aggregate_table(rows, save_dir / "experiment_summary.csv")
    plot_aggregate_summary(rows, save_dir / "experiment_summary.png")


if __name__ == "__main__":
    main()
