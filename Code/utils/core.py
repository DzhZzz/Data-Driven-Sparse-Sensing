"""Data-driven sparse sensing (DSS): basis, pivoted-QR placement, the linear DSS decoder and the ridge decoders."""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def build_basis(flows: np.ndarray, event_offsets: np.ndarray, weighted: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uncentered SVD basis of the development storms; returns the basis, the singular values s_k and the mode energies s_k^2 / N.

    weighted (the paper's basis): every storm block X_e is divided by ||X_e||_F, so that each storm contributes the same energy.
    Otherwise every snapshot has the same weight (the snapshot-equal comparison basis).
    """
    blocks = [flows[:, a:b] for a, b in zip(event_offsets[:-1], event_offsets[1:])]
    matrix = np.concatenate([x / np.sqrt(np.sum(x ** 2)) if weighted else x for x in blocks], axis=1)
    basis, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    return basis, singular_values, singular_values ** 2 / matrix.shape[1]


def sign_normalized(basis: np.ndarray) -> np.ndarray:
    """Columns signed so that their largest-magnitude entry is positive (for comparing bases only)."""
    rows = np.argmax(np.abs(basis), axis=0)
    signs = np.sign(basis[rows, np.arange(basis.shape[1])])
    signs[signs == 0.0] = 1.0
    return basis * signs


def qr_sensors(basis: np.ndarray, m: int) -> list[int]:
    """The first m pivots of the column-pivoted QR factorization of Psi_m^T (monitored conduits, in selection order)."""
    _, _, pivots = qr(basis[:, :m].T, pivoting=True, mode="economic")
    return [int(v) for v in pivots[:m]]


def decoder(basis: np.ndarray, sensors: list[int], rank: int) -> np.ndarray:
    """DSS decoder D = Psi_r (C Psi_r)^+, so that x_hat = D y for the flows y of the monitored conduits (no zero floor)."""
    modes = basis[:, :rank]
    return modes @ np.linalg.pinv(modes[sensors, :])


def condition_number(basis: np.ndarray, sensors: list[int]) -> float:
    """Condition number of the sampled basis C Psi_m (m = number of monitored conduits)."""
    return float(np.linalg.cond(basis[sensors, :len(sensors)]))


def ridge_decoder(flows: np.ndarray, event_offsets: np.ndarray, sensors: list[int], penalty: float, weighted: bool):
    """Ridge regression from the monitored flows to the flows of all 76 conduits, fitted on the development storms.

    Standardized predictors, unpenalized intercept, flows in physical units. weighted: every snapshot of storm e has the weight
    1 / ||X_e||_F^2 (normalized to a mean of one), as in the event-energy-weighted basis. Returns f with x_hat = f(y).
    """
    w = np.concatenate([np.full(b - a, 1.0 / np.sum(flows[:, a:b] ** 2) if weighted else 1.0)
                        for a, b in zip(event_offsets[:-1], event_offsets[1:])])
    w = w / w.mean()
    x, y = flows[sensors].T, flows.T
    x_mean, y_mean = np.average(x, axis=0, weights=w), np.average(y, axis=0, weights=w)
    x_std = np.sqrt(np.average((x - x_mean) ** 2, axis=0, weights=w))
    x_std[x_std < 1e-12] = 1.0
    z = (x - x_mean) / x_std
    u, s, vt = np.linalg.svd(z * np.sqrt(w)[:, None], full_matrices=False)
    uy = u.T @ ((y - y_mean) * np.sqrt(w)[:, None])
    if penalty == 0.0:
        gain = np.divide(1.0, s, out=np.zeros_like(s), where=s > np.finfo(float).eps * max(z.shape) * s[0])
    else:
        gain = s / (s ** 2 + w.sum() * penalty)
    coef = (vt.T @ (gain[:, None] * uy)) / x_std[:, None]
    intercept = y_mean - x_mean @ coef
    return lambda measured: (measured.T @ coef + intercept).T
