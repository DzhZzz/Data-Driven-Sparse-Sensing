"""Loading of the derived data in Data/ (conduit flow in m³/s)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from utils.paths import DATA_DIR

RANKS = tuple(range(1, 11))


def _npz(name: str) -> dict[str, np.ndarray]:
    with np.load(DATA_DIR / name, allow_pickle=False) as d:
        return {key: d[key] for key in d.files}


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, float_precision="round_trip", keep_default_na=False, na_values=[""])


def development() -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    """Development flows (76 x 5,824), conduit IDs, storm IDs and the snapshot offsets of the 17 development storms."""
    d = _npz("development_flows.npz")
    return (d["flows_m3s"].astype(float), [str(v) for v in d["link_ids"]], [str(v) for v in d["event_ids"]],
            d["event_offsets"].astype(int))


def heldout() -> tuple[np.ndarray, list[str], pd.DataFrame, np.ndarray]:
    """Held-out flows (76 x 59,450), conduit IDs, one row per simulation and the snapshot time stamps."""
    d = _npz("heldout_flows.npz")
    offsets = d["event_offsets"].astype(int)
    scenarios = pd.DataFrame({"scenario_id": d["scenario_ids"].astype(str), "parameter_set": d["parameter_set_ids"].astype(str),
                              "catalog_id": d["catalog_ids"].astype(str), "rainfall_type": d["event_types"].astype(str),
                              "start": offsets[:-1], "stop": offsets[1:]})
    return d["flows_m3s"].astype(float), [str(v) for v in d["link_ids"]], scenarios, d["timestamps"].astype(str)


def frozen_basis() -> dict[str, np.ndarray]:
    return _npz("frozen_basis.npz")


def layouts() -> dict[str, dict[int, list[int]]]:
    """Monitored conduits (rows of the state vector, in selection order) by method ('QR_DSS', 'GA_best', 'Area') and m."""
    table = _csv("sensor_layouts.csv")
    return {method: {int(row.m): [int(v) for v in str(row.sensor_indices).split(",")] for row in rows.itertuples()}
            for method, rows in table.groupby("method")}


@lru_cache(maxsize=1)
def ridge_penalties() -> pd.DataFrame:
    """Penalty of every ridge decoder, chosen by storm-wise cross-validation on the development storms."""
    return _csv("ridge_penalties.csv").fillna({"failed_link": ""})


def random_placement() -> pd.DataFrame:
    """Per-scenario system NSE of Uniform Random placement (median over 76 layouts at m = 1 and 2,048 at m = 2-10)."""
    return _csv("scenario_nse_random_placement.csv")


def measurement_errors() -> dict[str, np.ndarray]:
    """Per-scenario system NSE of QR-DSS under noise (median over 1,000 draws), detection floors and biases."""
    return _npz("scenario_nse_measurement_errors.npz")


def rainfall_catalog() -> pd.DataFrame:
    return _csv("rainfall_catalog.csv")


def calibration_series() -> pd.DataFrame:
    return _csv("calibration_series.csv")
