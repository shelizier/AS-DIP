"""Naming helpers for benchmark methods."""

from __future__ import annotations

METHOD_DISPLAY_NAMES = {
    "standard_dip": "Standard DIP",
    "drp_dip": "DRP-DIP",
    "as_dip": "AS-DIP",
}


def method_display_name(method: str) -> str:
    """Return a publication-friendly method name."""
    return METHOD_DISPLAY_NAMES.get(method, method.replace("_", " ").upper())
