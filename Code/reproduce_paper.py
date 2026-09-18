"""Reproduce manuscript Figures 3-10 from the compact derived data."""

from __future__ import annotations

import runpy
import sys

import numpy as np

from utils import data
from utils.core import build_direct_svd_basis, qr_sensor_indices
from utils.paths import CODE_DIR, OUTPUTS_DIR


OUT = OUTPUTS_DIR


def run_module(name: str, *arguments: str) -> None:
    previous = sys.argv
    try:
        sys.argv = [name, *arguments]
        runpy.run_module(name, run_name="__main__")
    finally:
        sys.argv = previous


def verify_core_method() -> None:
    training, node_ids, scale = data.load_training_matrix(CODE_DIR)
    basis, eigenvalues, _ = data.load_fixed_basis(CODE_DIR)
    rebuilt_basis, rebuilt_eigenvalues = build_direct_svd_basis(training, scale)
    if node_ids != data.load_testing_event_matrices(CODE_DIR)[1]:
        raise RuntimeError("Development and held-out node orders differ.")
    if not np.allclose(np.abs(rebuilt_basis), np.abs(basis), atol=2.0e-12, rtol=0.0):
        raise RuntimeError("Direct-SVD basis cannot be rebuilt from the compact development data.")
    if not np.allclose(rebuilt_eigenvalues, eigenvalues, atol=2.0e-15, rtol=0.0):
        raise RuntimeError("Direct-SVD spectrum cannot be rebuilt from the compact development data.")
    layouts = data.load_fixed_sensor_sets(CODE_DIR)
    for rank, frozen in layouts.items():
        if set(qr_sensor_indices(basis, rank)) != set(frozen):
            raise RuntimeError(f"Pivoted-QR sensor set drifted at r={rank}.")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.iterdir():
        if path.is_file():
            path.unlink()
    verify_core_method()
    run_module("figures.calibration")
    run_module("figures.reconstruction", "--figure", "all")
    run_module("figures.placement")
    run_module("figures.robustness", "--figure", "all")
    from verify_results import run as verify_results

    verify_results()
    print("PAPER_RESULTS_REPRODUCED")


if __name__ == "__main__":
    main()
