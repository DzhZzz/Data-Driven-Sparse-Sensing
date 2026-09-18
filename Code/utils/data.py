"""Compact, model-free data access for the paper reproduction package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.metrics import (
    STATE_VARIABLE_NODE_TOTAL_INFLOW,
    SYSTEM_LEVEL_NODE_SCOPE,
    SYSTEM_LEVEL_NSE_SCOPE_VERSION,
    system_level_nse,
)
from utils.paths import DATA_DIR, ROOT_DIR


CFS_TO_M3S = 0.028316846592
DEFAULT_STATE_VARIABLE = STATE_VARIABLE_NODE_TOTAL_INFLOW
DEFAULT_TESTING_SCOPE = "multi_model_rainfall"
NOMINAL_BASIS_WEIGHTING_SNAPSHOT_EQUAL = "snapshot_equal_direct_svd"
SPECTRAL_VALUE_KIND_UNCENTERED_SECOND_MOMENT_EIGENVALUE = "uncentered_second_moment_eigenvalue"
NODE_STD_THRESHOLD_M3S = 0.01
NODE_SENSITIVITY_THRESHOLDS_M3S = (0.0, 0.001, 0.005, 0.01, 0.02)
FORMAL_NODE_ACTIVITY_RULE = (
    "std(observed, ddof=0) >= 0.01 m3/s and nonzero NSE denominator"
)


def _root(code_dir: str | Path | None = None) -> Path:
    if code_dir is None:
        return ROOT_DIR
    path = Path(code_dir).resolve()
    return path.parent if path.name.lower() == "code" else path


def _data(code_dir: str | Path | None = None) -> Path:
    return DATA_DIR if code_dir is None else _root(code_dir) / "Data"


def normalize_state_variable(value: str | None = None) -> str:
    key = str(value or DEFAULT_STATE_VARIABLE).strip().lower().replace("-", "_")
    if key in {"flow", "inflow", "total_inflow", "node_total_flow"}:
        key = DEFAULT_STATE_VARIABLE
    if key != DEFAULT_STATE_VARIABLE:
        raise ValueError("This package supports node_total_inflow only.")
    return key


def normalize_testing_scope(value: str | None = None) -> str:
    key = str(value or DEFAULT_TESTING_SCOPE).strip().lower()
    aliases = {DEFAULT_TESTING_SCOPE, "multi", "compound", "compound_holdout"}
    if key not in aliases:
        raise ValueError("This package contains one held-out domain: multi_model_rainfall.")
    return DEFAULT_TESTING_SCOPE


def testing_scope_output_directory(scope: str | None = None) -> str:
    normalize_testing_scope(scope)
    return "Outputs"


def load_contract(
    code_dir: str | Path, state_variable: str = DEFAULT_STATE_VARIABLE
) -> dict[str, Any]:
    normalize_state_variable(state_variable)
    with np.load(_data(code_dir) / "frozen_basis.npz", allow_pickle=False) as payload:
        node_count = int(np.asarray(payload["nominal_basis"]).shape[0])
        scale = float(np.asarray(payload["normalization_scale_m3s"]).item())
        scenario_count = int(np.asarray(payload["nominal_training_scenario_count"]).item())
        snapshot_count = int(np.asarray(payload["nominal_training_snapshot_count"]).item())
        weighting = str(np.asarray(payload["nominal_basis_weighting"]).item())
        centering = str(np.asarray(payload["nominal_basis_centering"]).item())
    if (node_count, scenario_count, snapshot_count) != (77, 17, 5824):
        raise RuntimeError("Frozen development-domain dimensions are inconsistent.")
    if weighting != NOMINAL_BASIS_WEIGHTING_SNAPSHOT_EQUAL or centering != "none":
        raise RuntimeError("The frozen basis is not the paper's uncentered snapshot-equal SVD.")
    return {
        "state_variable": DEFAULT_STATE_VARIABLE,
        "node_count": node_count,
        "sensor_counts": list(range(1, 11)),
        "external_scenario_group": "COMPOUND_HOLDOUT",
        "external_scenario_count": 225,
        "nominal_training_scenario_count": scenario_count,
        "training_scenario_count": scenario_count,
        "nominal_training_snapshot_count": snapshot_count,
        "normalization_source_scenario_count": scenario_count,
        "normalization_source_snapshot_count": snapshot_count,
        "nominal_basis_weighting": weighting,
        "nominal_basis_centering": centering,
        "normalization_source": "nominal_17_event_max",
        "normalization_scale_m3s": scale,
        "training_global_max_m3s": scale,
    }


def load_fixed_basis(
    code_dir: str | Path, state_variable: str = DEFAULT_STATE_VARIABLE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalize_state_variable(state_variable)
    with np.load(_data(code_dir) / "frozen_basis.npz", allow_pickle=False) as payload:
        basis = np.asarray(payload["nominal_basis"], dtype=float)
        spectrum = np.asarray(payload["nominal_eigenvalues"], dtype=float)
    if basis.shape != (77, 77) or spectrum.shape != (77,):
        raise RuntimeError("Frozen basis dimensions are inconsistent.")
    return basis, spectrum, np.empty((77, 0), dtype=float)


def load_fixed_sensor_sets(
    code_dir: str | Path, state_variable: str = DEFAULT_STATE_VARIABLE
) -> dict[int, np.ndarray]:
    normalize_state_variable(state_variable)
    frame = pd.read_csv(_data(code_dir) / "sensor_layouts.csv", low_memory=False)
    frame = frame.loc[frame["Method"].astype(str).eq("NOMINAL_QR")]
    layouts: dict[int, np.ndarray] = {}
    for rank in range(1, 11):
        row = frame.loc[frame["Sensor_Count"].astype(int).eq(rank)]
        if len(row) != 1:
            raise RuntimeError(f"Expected one frozen QR layout at r={rank}.")
        values = np.asarray(
            [int(item) for item in str(row.iloc[0]["Sensor_Indices"]).split(",")],
            dtype=int,
        )
        if values.shape != (rank,):
            raise RuntimeError(f"Frozen QR layout at r={rank} has the wrong length.")
        layouts[rank] = values
    return layouts


def load_training_matrix(
    code_dir: str | Path, state_variable: str = DEFAULT_STATE_VARIABLE
) -> tuple[np.ndarray, list[str], float]:
    normalize_state_variable(state_variable)
    with np.load(_data(code_dir) / "development_flows.npz", allow_pickle=False) as payload:
        flows = np.asarray(payload["flows_cfs"], dtype=float) * CFS_TO_M3S
        node_ids = np.asarray(payload["node_ids"]).astype(str).tolist()
        offsets = np.asarray(payload["event_offsets"], dtype=np.int64)
        event_ids = np.asarray(payload["event_ids"]).astype(str)
    if flows.shape != (77, 5824) or offsets.shape != (18,) or len(event_ids) != 17:
        raise RuntimeError("Compact development bundle does not match the 17-event contract.")
    scale = float(load_contract(code_dir)["normalization_scale_m3s"])
    return flows, node_ids, scale


def load_testing_event_matrices(
    code_dir: str | Path,
    testing_scope: str = DEFAULT_TESTING_SCOPE,
    state_variable: str = DEFAULT_STATE_VARIABLE,
) -> tuple[list[dict[str, Any]], list[str]]:
    normalize_state_variable(state_variable)
    normalize_testing_scope(testing_scope)
    with np.load(_data(code_dir) / "heldout_flows.npz", allow_pickle=False) as payload:
        flows = np.asarray(payload["flows_cfs"], dtype=float) * CFS_TO_M3S
        node_ids = np.asarray(payload["node_ids"]).astype(str)
        scenario_ids = np.asarray(payload["scenario_ids"]).astype(str)
        parameter_ids = np.asarray(payload["parameter_set_ids"]).astype(str)
        catalog_ids = np.asarray(payload["catalog_ids"]).astype(str)
        event_types = np.asarray(payload["event_types"]).astype(str)
        rain_totals = np.asarray(payload["rain_total_mm"], dtype=float)
        offsets = np.asarray(payload["event_offsets"], dtype=np.int64)
        timestamps = np.asarray(payload["timestamps"]).astype(str)
    if len(scenario_ids) != 225 or offsets.shape != (226,) or flows.shape[0] != 77:
        raise RuntimeError("Compact held-out bundle does not match the 25 x 9 contract.")
    events: list[dict[str, Any]] = []
    for index, scenario_id in enumerate(scenario_ids):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        data = flows[:, start:stop]
        labels = timestamps[start:stop].tolist()
        events.append(
            {
                "event_name": scenario_id,
                "event_file": scenario_id,
                "event_type": event_types[index].lower(),
                "data": data,
                "event_matrix": data,
                "columns": labels,
                "time_labels": labels,
                "catalog_id": catalog_ids[index],
                "parameter_set_id": parameter_ids[index],
                "rain_total_mm": float(rain_totals[index]),
                "testing_scope": DEFAULT_TESTING_SCOPE,
                "state_variable": DEFAULT_STATE_VARIABLE,
            }
        )
    return events, node_ids.tolist()


def reconstruct_from_sensor_data(
    flow: np.ndarray,
    basis: np.ndarray,
    sensors: np.ndarray,
    rank: int,
    reconstruction_mode: str = "constrained",
) -> np.ndarray:
    mode = str(reconstruction_mode).strip().lower()
    if mode not in {"constrained", "unconstrained"}:
        raise ValueError("reconstruction_mode must be constrained or unconstrained")
    modes = np.asarray(basis[:, : int(rank)], dtype=float)
    operator = modes @ np.linalg.pinv(modes[np.asarray(sensors, dtype=int), :])
    prediction = operator @ np.asarray(flow[np.asarray(sensors, dtype=int), :], dtype=float)
    return np.maximum(prediction, 0.0) if mode == "constrained" else prediction


def validate_data_contract(code_dir: str | Path) -> dict[str, Any]:
    basis, _, _ = load_fixed_basis(code_dir)
    training, training_ids, scale = load_training_matrix(code_dir)
    events, heldout_ids = load_testing_event_matrices(code_dir)
    layouts = load_fixed_sensor_sets(code_dir)
    if training_ids != heldout_ids:
        raise RuntimeError("Development and held-out node order differs.")
    return {
        "status": "PASS",
        "basis_shape": list(basis.shape),
        "training_shape": list(training.shape),
        "training_global_max_m3s": scale,
        "heldout_scenarios": len(events),
        "heldout_parameter_vectors": len({item["parameter_set_id"] for item in events}),
        "heldout_rainfall_events": len({item["catalog_id"] for item in events}),
        "sensor_counts": sorted(layouts),
        "metric_scope": SYSTEM_LEVEL_NSE_SCOPE_VERSION,
        "node_scope": SYSTEM_LEVEL_NODE_SCOPE,
        "system_nse": system_level_nse,
    }
