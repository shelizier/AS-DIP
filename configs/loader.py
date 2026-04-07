"""YAML-aware argument loading for AS-DIP experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def _normalize_config_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    return {key.replace("-", "_"): value for key, value in config.items()}


def _load_yaml_config(config_path: str | None) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("YAML config must be a mapping of option names to values.")
    return _normalize_config_keys(config)


def build_parser(defaults: Dict[str, Any] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AS-DIP: Accelerated self-supervised seismic denoising.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML experiment configuration file.")
    parser.add_argument("--dataset-type", choices=["synthetic", "field"], default="synthetic")
    parser.add_argument("--mode", choices=["standard_dip", "drp_dip", "as_dip"], default="as_dip")
    parser.add_argument("--benchmark", action="store_true", help="Run Standard DIP, DRP-DIP, and AS-DIP benchmarks.")
    parser.add_argument("--backbone", choices=["unet", "lightweight"], default="unet")
    parser.add_argument("--activation", choices=["mish", "leaky_relu"], default="mish")
    parser.add_argument("--norm", choices=["batch", "instance"], default="batch")
    parser.add_argument("--iterations", type=int, default=2200)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-2,
        help="Fallback LR only; standard_dip / drp_dip / as_dip use fixed schedules in main.py.",
    )
    parser.add_argument(
        "--latent-learning-rate",
        type=float,
        default=1e-2,
        help="Fallback latent LR; main modes use fixed values (see main.build_trainer_config).",
    )
    parser.add_argument(
        "--standard-latent-learning-rate",
        type=float,
        default=5e-2,
        help="Latent LR for standard_dip only (DRP/AS-DIP use --latent-learning-rate).",
    )
    parser.add_argument("--adapter-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=0.05,
        help="Fallback TV weight; standard_dip / drp_dip / as_dip use fixed values in main.py.",
    )
    parser.add_argument("--tv-mode", choices=["l1", "l2"], default="l1")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--input-channels", type=int, default=32)
    parser.add_argument("--pad-border", type=int, default=0)
    parser.add_argument("--seed", type=int, default=121)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--experiment-name", type=str, default="as_dip_run")
    parser.add_argument("--noisy-path", type=str, default="data/field/noisy.npy")
    parser.add_argument("--clean-path", type=str, default="data/field/clean.npy")
    parser.add_argument("--disable-clean-reference", action="store_true")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--traces", type=int, default=128)
    parser.add_argument("--event-count", type=int, default=6)
    parser.add_argument("--gaussian-noise-std", type=float, default=0.12)
    parser.add_argument("--coherent-noise-amplitude", type=float, default=0.15)
    parser.add_argument("--coherent-slope", type=float, default=0.35)
    parser.add_argument("--save-inputs", action="store_true", help="Save noisy and clean input arrays in the run folder.")
    parser.set_defaults(**(defaults or {}))
    return parser


def parse_args(argv: list[str] | None = None) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args(argv)
    defaults = _load_yaml_config(bootstrap_args.config)
    parser = build_parser(defaults=defaults)
    args = parser.parse_args(argv)
    return args, defaults
