"""Dataset IO helpers for AS-DIP."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def load_npy(path: str | Path) -> np.ndarray:
    """Load a NumPy seismic section as float32."""
    return np.load(path).astype(np.float32)


def load_segy(path: str | Path) -> np.ndarray:
    """Load a SEG-Y seismic section if `segyio` is available."""
    try:
        import segyio
    except ImportError as exc:
        raise ImportError("Loading SEG-Y files requires the optional dependency `segyio`.") from exc

    with segyio.open(str(path), "r", ignore_geometry=True) as handle:
        data = segyio.tools.cube(handle)

    data = np.asarray(data, dtype=np.float32)
    while data.ndim > 2:
        data = data[0]
    return data


def load_field_data(noisy_path: str | Path, clean_path: Optional[str | Path] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load field or benchmark seismic data from NPY or SEG-Y formats."""
    noisy_path = Path(noisy_path)
    loader = load_segy if noisy_path.suffix.lower() in {".sgy", ".segy"} else load_npy
    noisy = loader(noisy_path)

    clean = None
    if clean_path is not None:
        clean_path = Path(clean_path)
        clean_loader = load_segy if clean_path.suffix.lower() in {".sgy", ".segy"} else load_npy
        clean = clean_loader(clean_path)
    return noisy, clean


def save_array(path: str | Path, array: np.ndarray) -> None:
    """Persist an array to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))
