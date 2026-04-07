# AS-DIP

English | [中文版](./README_zh.md)

Accelerated Seismic Deep Image Prior (AS-DIP) is a self-supervised seismic denoising framework that combines:

- Deep Image Prior (DIP) as an implicit structural regularizer
- Deep Random Projector (DRP) to reduce optimization cost by freezing most network weights
- Total Variation (TV) regularization to preserve seismic event continuity and suppress artifacts

The framework is designed for denoising seismic sections without requiring paired clean/noisy labels for training.

## Features

- Unified support for `standard_dip`, `drp_dip`, and `as_dip`
- Modular project structure for models, training, data, utilities, and scripts
- Synthetic seismic generation with Ricker wavelets, random noise, and coherent noise
- Field-data loading from `.npy` and optional SEG-Y formats
- Benchmarking utilities with automatic summary tables and comparison figures
- Publication-style plotting for seismic sections and residual analysis

## Project Structure

```text
AS-DIP/
├── configs/        # YAML experiment configs
├── core/           # Trainer, losses, device utilities
├── data/           # Synthetic generation and field data
│   └── field/      # Real seismic data used in this project
├── models/         # UNet, lightweight generators, DRP wrapper, activations
├── outputs/        # Experiment outputs
├── scripts/        # Benchmark and aggregation scripts
├── utils/          # Metrics, plotting, reporting, f-k tools
└── main.py         # Unified entry point
```

## Core Idea

AS-DIP takes a noisy seismic section and reconstructs it through an untrained neural network. Instead of learning from a dataset of clean/noisy pairs, it relies on network structure, a low-dimensional input seed, and seismic-specific regularization.

The repository currently supports three method variants:

1. `standard_dip`
   Optimizes the generator network parameters directly and serves as the classical DIP baseline.
2. `drp_dip`
   Keeps the original DRP-DIP idea as a comparison baseline by freezing most randomly initialized network weights and optimizing only the input seed and lightweight trainable layers.
3. `as_dip`
   Represents the proposed AS-DIP method. It keeps the DRP-style accelerated optimization strategy and adds TV regularization as a seismic-oriented enhancement.

In short:

- `standard_dip` = classical DIP baseline
- `drp_dip` = original DRP-DIP comparison method
- `as_dip` = proposed AS-DIP method

## Data

The field data currently used in this repository is stored at:

- `data/field/noisy.npy`
- `data/field/clean.npy`

These files were migrated from the legacy baseline folders and are now the main real-data source for experiments.

## Quick Start

Run a default experiment from YAML:

```bash
python main.py --config configs/default.yaml
```

The default config now runs `as_dip`.

Run a field-data benchmark:

```bash
python main.py \
  --dataset-type field \
  --benchmark \
  --iterations 10 \
  --experiment-name field_benchmark_full10 \
  --noisy-path data/field/noisy.npy \
  --clean-path data/field/clean.npy \
  --save-inputs
```

Run batch benchmarks from YAML:

```bash
python scripts/run_benchmark.py --config configs/benchmark_field.yaml
```

Aggregate all saved experiments:

```bash
python scripts/aggregate_results.py --outputs-dir outputs --save-dir outputs/aggregate
```

## Benchmark Methods

The current benchmark compares:

- `standard_dip`
- `drp_dip`
- `as_dip`

Displayed method names in reports are:

- Standard DIP
- DRP-DIP
- AS-DIP

Tracked metrics include:

- runtime
- PSNR
- SNR
- SNR gain
- SSIM
- residual energy

## Current Real-Data Result

The current saved full-size field benchmark output is stored in:

- `outputs/field_benchmark_full10/`

Important files include:

- `benchmark_summary.csv`
- `benchmark_summary.json`
- `method_overview.png`
- `benchmark_curves.png`
- `standard_dip/seismic_panels.png`
- `drp_dip/seismic_panels.png`
- `as_dip/seismic_panels.png`

Current benchmark summary on the field data:

- Standard DIP: `306.67 s`, `PSNR 13.20 dB`
- DRP-DIP: `199.63 s`, `PSNR 16.04 dB`
- AS-DIP: `207.94 s`, `PSNR 22.41 dB`

In the current 10-iteration field benchmark, AS-DIP produces the best reconstruction quality among the three compared methods.

## Notes

- The current field benchmark is a valid real-data baseline, but not yet a fully tuned final result.
- More iterations and hyperparameter search are still needed for publication-level performance.
- The repository no longer keeps the old `DIP/` and `DRP_DIP/` folders. Their useful contents have been reorganized into the current modular structure.

## Citation

If you use this repository in academic work, please cite the original DIP and DRP ideas as well as any seismic denoising references relevant to your final implementation.
