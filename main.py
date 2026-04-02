"""Unified entry point for AS-DIP seismic denoising experiments."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl_cache"))

from configs import parse_args
from core.device import seed_everything, select_device
from core.trainer import ASDIPTrainer, TrainerConfig
from data.io import load_field_data, save_array
from data.synthetic import SyntheticSeismicConfig, create_synthetic_seismic_sample
from utils.naming import method_display_name
from utils.plotting import plot_benchmark_curves, plot_seismic_panels
from utils.reporting import export_benchmark_summary, plot_method_overview


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def build_trainer_config(args: argparse.Namespace, mode: str) -> TrainerConfig:
    return TrainerConfig(
        mode=mode,
        backbone=args.backbone,
        input_channels=args.input_channels,
        activation=args.activation,
        norm=args.norm,
        learning_rate=args.learning_rate,
        seed_learning_rate=args.seed_learning_rate,
        adapter_learning_rate=args.adapter_learning_rate,
        iterations=args.iterations,
        tv_weight=args.tv_weight,
        tv_mode=args.tv_mode,
        log_interval=args.log_interval,
        pad_border=args.pad_border,
        reg_noise_std=0.03 if mode == "standard_dip" else 0.0,
    )


def prepare_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray | None, Dict[str, np.ndarray]]:
    if args.dataset_type == "synthetic":
        config = SyntheticSeismicConfig(
            samples=args.samples,
            traces=args.traces,
            event_count=args.event_count,
            gaussian_noise_std=args.gaussian_noise_std,
            coherent_noise_amplitude=args.coherent_noise_amplitude,
            coherent_slope=args.coherent_slope,
            seed=args.seed,
        )
        sample = create_synthetic_seismic_sample(config)
        return sample["noisy"], sample["clean"], sample

    clean_path = None if args.disable_clean_reference else args.clean_path
    noisy, clean = load_field_data(args.noisy_path, clean_path=clean_path)
    return noisy, clean, {"noisy": noisy, "clean": clean if clean is not None else noisy}


def summarize_result(name: str, result: Any) -> str:
    metrics = result.best_metrics
    display_name = method_display_name(name)
    if not metrics:
        return f"{display_name}: time={result.elapsed_seconds:.2f}s, best_iter={result.best_iteration}"
    summary = (
        f"{display_name}: time={result.elapsed_seconds:.2f}s, best_iter={result.best_iteration}, "
        f"SNR gain={metrics.get('snr_gain', float('nan')):.2f} dB"
    )
    if "psnr" in metrics:
        summary += f", PSNR={metrics['psnr']:.2f} dB"
    return summary


def run_experiment(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(prefer_cuda=True)
    logging.info("Using device: %s", device)

    noisy, clean, metadata = prepare_dataset(args)
    output_root = Path(args.output_dir) / args.experiment_name
    output_root.mkdir(parents=True, exist_ok=True)
    if getattr(args, "save_inputs", False):
        save_array(output_root / "input_noisy.npy", noisy)
        if clean is not None:
            save_array(output_root / "input_clean.npy", clean)

    modes = ["standard_dip", "drp_dip", "as_dip"] if args.benchmark else [args.mode]
    benchmark_histories: Dict[str, Dict[str, list[float]]] = {}
    results = {}

    for mode in modes:
        logging.info("Starting experiment mode=%s", method_display_name(mode))
        mode_dir = output_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        trainer = ASDIPTrainer(build_trainer_config(args, mode), device=device)
        result = trainer.run(noisy=noisy, clean=clean, output_dir=mode_dir)
        results[mode] = result
        benchmark_histories[mode] = result.history

        save_array(mode_dir / "denoised.npy", result.denoised)
        save_array(mode_dir / "residual.npy", result.residual)
        plot_seismic_panels(
            original=metadata["clean"] if metadata.get("clean") is not None else noisy,
            noisy=noisy,
            denoised=result.denoised,
            residual=result.residual,
            clean=clean,
            save_path=mode_dir / "seismic_panels.png",
        )
        logging.info(summarize_result(mode, result))

    plot_method_overview(
        noisy=noisy,
        clean=clean,
        results=results,
        save_path=output_root / "method_overview.png",
    )
    export_benchmark_summary(
        output_root=output_root,
        experiment_name=args.experiment_name,
        dataset_type=args.dataset_type,
        results=results,
    )

    if len(benchmark_histories) > 1:
        plot_benchmark_curves(benchmark_histories, output_root / "benchmark_curves.png")

    if clean is not None and len(results) > 1:
        standard = results.get("standard_dip")
        drp = results.get("drp_dip")
        as_dip = results.get("as_dip")
        if standard is not None and drp is not None:
            logging.info(
                "Acceleration summary | Standard DIP=%.2fs | DRP-DIP=%.2fs | speedup=%.2fx",
                standard.elapsed_seconds,
                drp.elapsed_seconds,
                standard.elapsed_seconds / max(drp.elapsed_seconds, 1e-6),
            )
        if standard is not None and as_dip is not None:
            logging.info(
                "Acceleration summary | Standard DIP=%.2fs | AS-DIP=%.2fs | speedup=%.2fx",
                standard.elapsed_seconds,
                as_dip.elapsed_seconds,
                standard.elapsed_seconds / max(as_dip.elapsed_seconds, 1e-6),
            )


if __name__ == "__main__":
    setup_logging()
    parsed_args, _ = parse_args()
    run_experiment(parsed_args)
