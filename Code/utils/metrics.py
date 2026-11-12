"""Nash-Sutcliffe efficiency as used in the paper."""

from __future__ import annotations

import numpy as np

ACTIVE_STD_M3S = 0.01  # a conduit is active (evaluable) in a simulation when the std of its simulated flow reaches 0.01 m³/s
EPSILON = 1.0e-15


def system_nse(x: np.ndarray, x_hat: np.ndarray) -> float:
    """System-level NSE: all conduits and time steps of one simulation pooled, with one mean."""
    return float(1.0 - np.sum((x - x_hat) ** 2) / np.sum((x - x.mean()) ** 2))


def conduit_nse(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """NSE of every conduit (row); NaN where the simulated series is constant."""
    denominator = np.sum((x - x.mean(axis=1, keepdims=True)) ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > EPSILON, 1.0 - np.sum((x - x_hat) ** 2, axis=1) / denominator, np.nan)
