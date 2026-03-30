"""Synthetic 2D seismic section generation for AS-DIP experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticSeismicConfig:
    """Synthetic seismic profile settings."""

    samples: int = 256
    traces: int = 128
    event_count: int = 6
    wavelet_frequency: float = 25.0
    wavelet_length: float = 0.128
    dt: float = 0.004
    gaussian_noise_std: float = 0.12
    coherent_noise_amplitude: float = 0.15
    coherent_slope: float = 0.35
    seed: int = 42


def ricker_wavelet(frequency: float, length: float, dt: float) -> np.ndarray:
    """Generate a Ricker wavelet."""
    time = np.arange(-length / 2.0, length / 2.0 + dt, dt, dtype=np.float32)
    argument = (np.pi * frequency * time) ** 2
    return (1.0 - 2.0 * argument) * np.exp(-argument)


def _build_reflectivity(config: SyntheticSeismicConfig, rng: np.random.Generator) -> np.ndarray:
    reflectivity = np.zeros((config.samples, config.traces), dtype=np.float32)
    trace_indices = np.arange(config.traces, dtype=np.float32)
    margin = max(4, min(20, config.samples // 6))
    low = margin
    high = max(low + 1, config.samples - margin)
    for event_idx in range(config.event_count):
        center = rng.integers(low=low, high=high)
        slope = rng.uniform(-0.25, 0.25)
        curvature = rng.uniform(-0.0008, 0.0008)
        amplitude = rng.uniform(0.6, 1.0) * (1.0 if event_idx % 2 == 0 else -1.0)
        arrival = center + slope * (trace_indices - config.traces / 2.0) + curvature * (trace_indices - config.traces / 2.0) ** 2
        arrival = np.clip(np.round(arrival).astype(int), 0, config.samples - 1)
        reflectivity[arrival, np.arange(config.traces)] += amplitude
    return reflectivity


def _convolve_traces(reflectivity: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
    section = np.zeros_like(reflectivity)
    for trace_idx in range(reflectivity.shape[1]):
        convolved = np.convolve(reflectivity[:, trace_idx], wavelet, mode="full")
        start = (len(convolved) - reflectivity.shape[0]) // 2
        stop = start + reflectivity.shape[0]
        section[:, trace_idx] = convolved[start:stop]
    return section.astype(np.float32)


def _coherent_noise(config: SyntheticSeismicConfig, rng: np.random.Generator) -> np.ndarray:
    noise = np.zeros((config.samples, config.traces), dtype=np.float32)
    time = np.arange(config.samples, dtype=np.float32)
    for intercept in np.linspace(0.15, 0.85, 3):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        center = intercept * config.samples
        for trace_idx in range(config.traces):
            shifted_time = time - (center + config.coherent_slope * trace_idx)
            envelope = np.exp(-0.5 * (shifted_time / 14.0) ** 2)
            carrier = np.sin(2.0 * np.pi * 0.035 * shifted_time + phase)
            noise[:, trace_idx] += config.coherent_noise_amplitude * envelope * carrier
    return noise


def normalize_amplitude(section: np.ndarray) -> np.ndarray:
    """Normalize a section to symmetric range [-1, 1]."""
    max_abs = np.max(np.abs(section))
    if max_abs < 1e-8:
        return section.astype(np.float32)
    return (section / max_abs).astype(np.float32)


def create_synthetic_seismic_sample(config: SyntheticSeismicConfig) -> dict[str, np.ndarray]:
    """Create clean, noisy, and component arrays for benchmarking."""
    rng = np.random.default_rng(config.seed)
    wavelet = ricker_wavelet(config.wavelet_frequency, config.wavelet_length, config.dt)
    reflectivity = _build_reflectivity(config, rng)
    clean = _convolve_traces(reflectivity, wavelet)
    gaussian_noise = rng.normal(0.0, config.gaussian_noise_std, size=clean.shape).astype(np.float32)
    coherent_noise = _coherent_noise(config, rng)
    noisy = clean + gaussian_noise + coherent_noise

    clean = normalize_amplitude(clean)
    gaussian_noise = normalize_amplitude(gaussian_noise)
    coherent_noise = normalize_amplitude(coherent_noise)
    noisy = normalize_amplitude(noisy)

    return {
        "clean": clean,
        "noisy": noisy,
        "gaussian_noise": gaussian_noise,
        "coherent_noise": coherent_noise,
        "reflectivity": reflectivity.astype(np.float32),
        "wavelet": wavelet.astype(np.float32),
    }
