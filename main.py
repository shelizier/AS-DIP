from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import logging
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
    # 只有原始 DRP 强制用 batch norm，让 AS-DIP 使用更高级的 norm (默认 instance)
    use_paper_style_drp = mode == "drp_dip"
    use_asdip_enhancement = mode == "as_dip"

    # 默认参数打底
    exp_smoothing = 0.99
    phase1_fraction = 0.2
    phase2_fraction = 0.3
    tv_weight_t = 0.1
    tv_weight_x = 1.0
    # 默认的相似度阈值
    residual_sim_threshold = 0.95

    if mode == "standard_dip":
        learning_rate = 1e-3
        latent_learning_rate = args.standard_latent_learning_rate
        # 【公平起见】：给 Standard DIP 也加上一点 TV 参与神仙打架！
        tv_weight = 0.02
        reg_noise_std = 0.03
    elif mode == "drp_dip":
        learning_rate = 0.1
        latent_learning_rate = 0.1
        tv_weight = 0.25  # DRP 必须高 TV
        reg_noise_std = 0.03
    elif mode == "as_dip":
        learning_rate = 0.1
        latent_learning_rate = 0.01

        # 使用 getattr 安全获取，如果 loader.py 没注册这个参数，就默认用后面的数值
        tv_weight = getattr(args, "tv_weight", 0.15)
        reg_noise_std = getattr(args, "reg_noise_std", 0.015)

        use_asdip_enhancement = True
        exp_smoothing = 0.95
        phase1_fraction = 0.15
        phase2_fraction = 0.15

        tv_weight_t = getattr(args, "tv_weight_t", 0.1)
        tv_weight_x = getattr(args, "tv_weight_x", 1.5)
        residual_sim_threshold = getattr(args, "residual_similarity_threshold", 0.95)
    else:
        learning_rate = args.learning_rate
        latent_learning_rate = args.latent_learning_rate
        tv_weight = args.tv_weight
        reg_noise_std = 0.0

    current_iterations = args.iterations if mode == "standard_dip" else int(args.iterations * 1.5)

    return TrainerConfig(
        mode=mode,
        backbone=args.backbone,
        input_channels=args.input_channels,
        activation=args.activation,
        norm="batch" if use_paper_style_drp else args.norm,
        learning_rate=learning_rate,
        latent_learning_rate=latent_learning_rate,
        adapter_learning_rate=args.adapter_learning_rate,
        iterations=current_iterations,
        tv_weight=tv_weight,
        tv_mode=args.tv_mode,
        log_interval=args.log_interval,
        pad_border=args.pad_border,
        reg_noise_std=reg_noise_std,
        reg_noise_std_phase1=(0.035 if use_asdip_enhancement else None),
        reg_noise_std_phase2=(0.03 if use_asdip_enhancement else None),
        reg_noise_std_phase3=(0.022 if use_asdip_enhancement else None),
        train_output_adapter=(mode == "as_dip"),
        phased_optimization=(mode == "as_dip"),
        backbone_learning_rate=min(learning_rate, args.adapter_learning_rate) * 0.5,
        use_structured_latent=use_asdip_enhancement,
        latent_coord_channels=2,
        mse_weight=1.0,
        l1_weight=0.1,
        gradient_weight=0.0,
        phase3_latent_lr_scale=0.5 if use_asdip_enhancement else 0.1,

        tv_weight_t=tv_weight_t,
        tv_weight_x=tv_weight_x,
        phase1_fraction=phase1_fraction,
        phase2_fraction=phase2_fraction,
        exp_smoothing=exp_smoothing,
        # 传入解除封印的阈值
        residual_similarity_threshold=residual_sim_threshold,
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
        f"SNR={metrics.get('snr', float('nan')):.2f} dB, "
        f"SNR gain={metrics.get('snr_gain', float('nan')):.2f} dB"
    )
    return summary


def run_experiment(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device()
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