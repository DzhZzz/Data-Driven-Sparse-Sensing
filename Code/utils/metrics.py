"""Metric definitions used by the first-paper reconstruction figures.

The paper evaluates model-conditioned node-total-inflow states.  The package
intentionally exposes one state variable and one system-level NSE scope so that
the figure scripts cannot silently switch to an older variable or metric.
"""

from __future__ import annotations

import numpy as np


STATE_VARIABLE_NODE_TOTAL_INFLOW = "node_total_inflow"
SYSTEM_LEVEL_NSE_SCOPE_VERSION = "all_nodes_all_available_time_steps_v1"
SYSTEM_LEVEL_NODE_SCOPE = "all_77_nodes_all_available_time_steps"


def normalize_metric_state_variable(value: str | None) -> str:
    key = str(value or STATE_VARIABLE_NODE_TOTAL_INFLOW).strip().lower().replace("-", "_")
    aliases = {"flow": STATE_VARIABLE_NODE_TOTAL_INFLOW, "inflow": STATE_VARIABLE_NODE_TOTAL_INFLOW,
               "total_inflow": STATE_VARIABLE_NODE_TOTAL_INFLOW}
    normalized = aliases.get(key, key)
    if normalized != STATE_VARIABLE_NODE_TOTAL_INFLOW:
        raise ValueError("This paper package supports node_total_inflow only.")
    return normalized


def nse_denominator(observed: np.ndarray) -> float:
    values = np.asarray(observed, dtype=float)
    return float(np.sum((values - float(np.mean(values))) ** 2))


def system_level_nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape:
        raise ValueError(f"NSE shape mismatch: {observed.shape} != {predicted.shape}")
    denominator = nse_denominator(observed)
    if denominator <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((observed - predicted) ** 2) / denominator)


def metric_spec(value: str | None = None) -> dict[str, str]:
    normalize_metric_state_variable(value)
    return {
        "state_variable": STATE_VARIABLE_NODE_TOTAL_INFLOW,
        "scope_version": SYSTEM_LEVEL_NSE_SCOPE_VERSION,
        "node_scope": SYSTEM_LEVEL_NODE_SCOPE,
        "unit": "m³/s",
    }
