"""Run one or more AS-DIP benchmark jobs from a YAML file."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AS-DIP benchmark batches from YAML.")
    parser.add_argument("--config", required=True, help="Benchmark YAML containing an `experiments` list.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    experiments = config.get("experiments", [])
    if not experiments:
        raise ValueError("Benchmark config must define a non-empty `experiments` list.")

    main_script = PROJECT_ROOT / "main.py"

    for experiment in experiments:
        command = [sys.executable, str(main_script)]
        for key, value in experiment.items():
            option = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    command.append(option)
                continue
            command.extend([option, str(value)])
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
