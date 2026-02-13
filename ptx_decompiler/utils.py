"""Shared utilities for device detection and environment helpers."""

import torch


def get_device() -> torch.device:
    """
    Return best available device: cuda > mps > cpu.

    - CUDA: NVIDIA GPUs (Colab T4/A100, desktop RTX, etc.)
    - MPS:  Apple Silicon (M1/M2/M3 Mac) via Metal Performance Shaders
    - CPU:  Fallback
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def supports_amp(device: torch.device) -> bool:
    """Check if the device supports torch.amp mixed-precision training."""
    # CUDA supports full AMP (autocast + GradScaler).
    # MPS has partial autocast but no GradScaler — safer to disable.
    return device.type == "cuda"


def is_colab() -> bool:
    """Detect if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False
