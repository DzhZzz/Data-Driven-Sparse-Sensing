"""Core direct-SVD sparse sensing operations used in the paper."""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def build_direct_svd_basis(
    development_flows_m3s: np.ndarray,
    normalization_scale_m3s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the uncentered, snapshot-equal spatial basis.

    Columns are five-minute snapshots. No event weighting or centering is
    applied; every development snapshot contributes once.
    """

    matrix = np.asarray(development_flows_m3s, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 77:
        raise ValueError("Expected a 77 x N development-flow matrix.")
    if not np.isfinite(normalization_scale_m3s) or normalization_scale_m3s <= 0:
        raise ValueError("normalization_scale_m3s must be positive.")
    basis, singular_values, _ = np.linalg.svd(
        matrix / float(normalization_scale_m3s), full_matrices=False
    )
    eigenvalues = singular_values**2 / matrix.shape[1]
    return basis, eigenvalues


def qr_sensor_indices(basis: np.ndarray, rank: int) -> np.ndarray:
    """Select `rank` node indices by pivoted QR of the retained modes."""

    modes = np.asarray(basis, dtype=float)[:, : int(rank)]
    if modes.shape != (77, int(rank)) or rank < 1:
        raise ValueError("rank must be between 1 and the available mode count.")
    _, _, pivots = qr(modes.T, pivoting=True, mode="economic")
    return np.asarray(pivots[:rank], dtype=int)


def reconstruct_flows(
    basis: np.ndarray,
    sensor_indices: np.ndarray,
    sensor_measurements: np.ndarray,
    rank: int,
    nonnegative: bool = True,
) -> np.ndarray:
    """Reconstruct all node flows from the sampled rows of the retained basis."""

    modes = np.asarray(basis, dtype=float)[:, : int(rank)]
    sensors = np.asarray(sensor_indices, dtype=int)
    measurements = np.asarray(sensor_measurements, dtype=float)
    coefficients = np.linalg.pinv(modes[sensors, :]) @ measurements
    reconstructed = modes @ coefficients
    return np.maximum(reconstructed, 0.0) if nonnegative else reconstructed
