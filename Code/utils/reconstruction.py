"""Load the frozen experiment context and compute reconstruction metrics."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from utils.data import (
    DEFAULT_STATE_VARIABLE,
    NODE_STD_THRESHOLD_M3S,
    load_contract,
    load_fixed_basis,
    load_fixed_sensor_sets,
    load_testing_event_matrices,
    load_training_matrix,
)
from utils.metrics import system_level_nse
from utils.paths import CODE_DIR


DEFAULT_RANKS = tuple(range(1, 11))
NSE_EPSILON = 1.0e-15


def normalize_node_id(value: object) -> str:
    return re.sub(r"(OF)(\d+)", r"OF-\2", str(value))


def finite_nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    denominator = float(np.sum((observed - np.mean(observed)) ** 2))
    if denominator <= NSE_EPSILON or not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
        return float("nan")
    return float(1.0 - np.sum((predicted - observed) ** 2) / denominator)


def node_is_evaluable(observed: np.ndarray) -> bool:
    values = np.asarray(observed, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return False
    return bool(np.std(values, ddof=0) >= NODE_STD_THRESHOLD_M3S and
                np.sum((values - np.mean(values)) ** 2) > NSE_EPSILON)


def load_context() -> dict[str, Any]:
    contract = load_contract(CODE_DIR, DEFAULT_STATE_VARIABLE)
    basis, eigenvalues, _ = load_fixed_basis(CODE_DIR, DEFAULT_STATE_VARIABLE)
    sensor_sets = load_fixed_sensor_sets(CODE_DIR, DEFAULT_STATE_VARIABLE)
    _, node_ids, scale = load_training_matrix(CODE_DIR, DEFAULT_STATE_VARIABLE)
    events, event_node_ids = load_testing_event_matrices(
        CODE_DIR,
        testing_scope="multi_model_rainfall",
        state_variable=DEFAULT_STATE_VARIABLE,
    )
    if node_ids != event_node_ids:
        raise RuntimeError("Training and held-out node ordering differ.")
    if len(events) != 225 or len(node_ids) != 77:
        raise RuntimeError(f"Expected 225 held-out scenarios and 77 nodes; got {len(events)}, {len(node_ids)}")
    if sorted(sensor_sets) != list(DEFAULT_RANKS):
        raise RuntimeError(f"Frozen QR layouts must cover r=1..10; got {sorted(sensor_sets)}")
    return {
        "contract": contract,
        "basis": np.asarray(basis, dtype=float),
        "eigenvalues": np.asarray(eigenvalues, dtype=float),
        "sensor_sets": sensor_sets,
        "node_ids": [normalize_node_id(value) for value in node_ids],
        "scale": float(scale),
        "events": events,
    }


def projection_operators(basis: np.ndarray, sensor_sets: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    operators: dict[int, np.ndarray] = {}
    for rank, sensors in sensor_sets.items():
        basis_r = np.asarray(basis[:, :rank], dtype=float)
        operators[int(rank)] = basis_r @ np.linalg.pinv(basis_r[np.asarray(sensors, dtype=int), :])
    return operators


def reconstruct(event_data: np.ndarray, operator: np.ndarray, sensors: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.asarray(event_data, dtype=float) / float(scale)
    reconstructed = operator @ normalized[np.asarray(sensors, dtype=int), :]
    return np.maximum(reconstructed, 0.0) * float(scale)


def calculate_metrics(context: dict[str, Any], ranks: tuple[int, ...] = DEFAULT_RANKS) -> dict[str, Any]:
    """Calculate system and node NSE from the same frozen reconstruction path."""

    events = context["events"]
    basis = context["basis"]
    sensors = context["sensor_sets"]
    node_ids = context["node_ids"]
    operators = projection_operators(basis, sensors)
    system: dict[str, dict[int, list[float]]] = {
        "observed": {rank: [] for rank in ranks},
        "design": {rank: [] for rank in ranks},
    }
    system_rows: dict[str, list[dict[str, Any]]] = {"observed": [], "design": []}
    node_lists: dict[str, dict[int, dict[str, list[float]]]] = {
        "observed": {rank: {node: [] for node in node_ids} for rank in ranks},
        "design": {rank: {node: [] for node in node_ids} for rank in ranks},
    }
    node_event_maps: dict[str, dict[int, list[dict[str, float]]]] = {
        "observed": {rank: [] for rank in ranks},
        "design": {rank: [] for rank in ranks},
    }
    network_medians: dict[str, dict[int, list[float]]] = {
        "observed": {rank: [] for rank in ranks},
        "design": {rank: [] for rank in ranks},
    }

    for event in events:
        event_type = str(event.get("event_type", "")).strip().lower()
        if event_type not in system:
            continue
        observed = np.asarray(event["data"], dtype=float) * 1.0
        for rank in ranks:
            predicted = reconstruct(observed, operators[rank], sensors[rank], context["scale"])
            overall = float(system_level_nse(observed, predicted))
            if not np.isfinite(overall):
                overall = finite_nse(observed, predicted)
            system[event_type][rank].append(overall)
            system_rows[event_type].append({
                "Sensor_Count": int(rank),
                "Event_Name": str(event["event_name"]),
                "Overall_NSE": overall,
            })
            event_node_map: dict[str, float] = {}
            for index, node_id in enumerate(node_ids):
                if not node_is_evaluable(observed[index]):
                    continue
                value = finite_nse(observed[index], predicted[index])
                if np.isfinite(value):
                    node_lists[event_type][rank][node_id].append(value)
                    event_node_map[node_id] = value
            node_event_maps[event_type][rank].append(event_node_map)
            if event_node_map:
                network_medians[event_type][rank].append(float(np.median(list(event_node_map.values()))))

    node_means: dict[str, dict[int, dict[str, float]]] = {"observed": {}, "design": {}, "all": {}}
    for event_type in ("observed", "design"):
        for rank in ranks:
            node_means[event_type][rank] = {
                node: float(np.mean(values))
                for node, values in node_lists[event_type][rank].items()
                if values
            }
    for rank in ranks:
        node_means["all"][rank] = {}
        for node in node_ids:
            values = node_lists["observed"][rank][node] + node_lists["design"][rank][node]
            if values:
                node_means["all"][rank][node] = float(np.mean(values))
    return {
        "system": system,
        "system_rows": system_rows,
        "node_lists": node_lists,
        "node_means": node_means,
        "node_event_maps": node_event_maps,
        "network_medians": network_medians,
    }
