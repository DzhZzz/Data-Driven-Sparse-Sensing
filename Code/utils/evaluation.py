"""Development and held-out evaluation: basis checks, decoders, system and conduit metrics, sensor loss."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import data
from utils.core import build_basis, decoder, qr_sensors, ridge_decoder, sign_normalized
from utils.metrics import ACTIVE_STD_M3S, conduit_nse, system_nse


def context() -> dict:
    """Basis and QR layouts rebuilt from the development storms and Area layouts from the upstream areas (all checked against Data/),
    and the held-out data."""
    dev, links, events, offsets = data.development()
    frozen = data.frozen_basis()
    basis, singular_values, energies = build_basis(dev, offsets)
    assert np.max(np.abs(sign_normalized(basis) - sign_normalized(frozen["nominal_basis"]))) < 1e-12, "basis not reproduced"
    assert np.max(np.abs(singular_values - frozen["nominal_singular_values"])) < 1e-12
    flows, held_links, scenarios, timestamps = data.heldout()
    assert held_links == links
    layouts = data.layouts()
    qr = {m: qr_sensors(basis, m) for m in data.RANKS}
    assert qr == layouts["QR_DSS"], "QR layouts not reproduced"
    return {"dev": dev, "events": events, "offsets": offsets, "basis": basis, "energies": energies, "links": links, "flows": flows,
            "scenarios": scenarios, "timestamps": timestamps, "layouts": layouts}


def blocks(ctx: dict):
    """(scenario row, flows of that simulation) for the 225 held-out simulations."""
    for row in ctx["scenarios"].itertuples():
        yield row, ctx["flows"][:, row.start:row.stop]


def dss(ctx: dict, sensors: list[int], rank: int | None = None):
    d = decoder(ctx["basis"], sensors, len(sensors) if rank is None else rank)
    return lambda measured: d @ measured


def ridge(ctx: dict, sites: str, m: int, weighted: bool, failed_link: str = ""):
    """Ridge decoder at the m QR or Area sites (without `failed_link`); penalty chosen by cross-validation (Data/ridge_penalties.csv)."""
    sensors = [s for s in ctx["layouts"]["QR_DSS" if sites == "QR" else "Area"][m] if ctx["links"][s] != failed_link]
    p = data.ridge_penalties()
    row = p[(p.decoder == ("weighted_ridge" if weighted else "ridge")) & (p.sites == sites) & (p.m == m) & (p.failed_link == failed_link)]
    assert len(row) == 1, (sites, m, failed_link, weighted)
    return sensors, ridge_decoder(ctx["dev"], ctx["offsets"], sensors, float(row["lambda"].iloc[0]), weighted)


def models(ctx: dict) -> dict[str, dict[int, tuple[list[int], object]]]:
    """Decoders compared in the paper, by method and m: (monitored conduits, x_hat = f(y))."""
    lay = ctx["layouts"]
    out = {"QR_DSS": {}, "GA_best_DSS": {}, "Area_DSS": {}, "QR_Ridge": {}, "QR_RidgeW": {}, "Area_RidgeW": {}}
    for m in data.RANKS:
        out["QR_DSS"][m] = (lay["QR_DSS"][m], dss(ctx, lay["QR_DSS"][m]))
        out["GA_best_DSS"][m] = (lay["GA_best"][m], dss(ctx, lay["GA_best"][m]))
        out["Area_DSS"][m] = (lay["Area"][m], dss(ctx, lay["Area"][m]))
        out["QR_Ridge"][m] = ridge(ctx, "QR", m, weighted=False)
        out["QR_RidgeW"][m] = ridge(ctx, "QR", m, weighted=True)
        out["Area_RidgeW"][m] = ridge(ctx, "Area", m, weighted=True)
    return out


def system_nse_table(ctx: dict, by_method: dict) -> pd.DataFrame:
    """Per-scenario system NSE for every method and m of `by_method` (as returned by `models`)."""
    rows = []
    for row, x in blocks(ctx):
        for method, by_m in by_method.items():
            for m, (sensors, f) in by_m.items():
                rows.append((method, m, row.scenario_id, row.catalog_id, row.rainfall_type, system_nse(x, f(x[sensors]))))
    return pd.DataFrame(rows, columns=["method", "m", "scenario_id", "catalog_id", "rainfall_type", "system_nse"])


def conduit_metrics(ctx: dict, sensors: list[int], f) -> dict[str, np.ndarray]:
    """NSE, mean absolute error and temporal std (m³/s) of every conduit in every held-out simulation (225 x 76)."""
    nse, mae, std = [], [], []
    for _, x in blocks(ctx):
        x_hat = f(x[sensors])
        nse.append(conduit_nse(x, x_hat))
        mae.append(np.mean(np.abs(x_hat - x), axis=1))
        std.append(np.std(x, axis=1))
    return {"nse": np.array(nse), "mae": np.array(mae), "std": np.array(std), "active": np.array(std) >= ACTIVE_STD_M3S, "sensors": sensors}


def sensor_loss(ctx: dict) -> pd.DataFrame:
    """System NSE of the complete QR layout minus that after the failure of one monitored conduit, m = 2-10, for three decoders of
    the m - 1 remaining conduits: uncompensated (rank-m DSS decoder), truncated (rank m - 1) and refit (weighted ridge)."""
    rows = []
    for m in range(2, 11):
        sensors = ctx["layouts"]["QR_DSS"][m]
        full = dss(ctx, sensors)
        for k, failed in enumerate(sensors):
            rest = [s for i, s in enumerate(sensors) if i != k]
            link = ctx["links"][failed]
            variants = {"uncompensated": dss(ctx, rest, m), "truncated": dss(ctx, rest, m - 1),
                        "refit": ridge(ctx, "QR", m, weighted=True, failed_link=link)[1]}
            for row, x in blocks(ctx):
                reference = system_nse(x, full(x[sensors]))
                for variant, f in variants.items():
                    rows.append((variant, m, link, row.scenario_id, row.rainfall_type, reference - system_nse(x, f(x[rest]))))
    return pd.DataFrame(rows, columns=["variant", "m", "failed_link", "scenario_id", "rainfall_type", "loss"])


def basis_weighting_cv(ctx: dict) -> dict[str, float]:
    """Storm-wise cross-validation on the development storms (each observed storm alone, the five design storms together): mean
    over m = 1-10 of the mean fold score, the fold score being the mean system NSE of its storms, clipped below at -1."""
    blocks_ = [ctx["dev"][:, a:b] for a, b in zip(ctx["offsets"][:-1], ctx["offsets"][1:])]
    design = [i for i, e in enumerate(ctx["events"]) if e.startswith("DESIGN")]
    folds = [[i] for i in range(len(blocks_)) if i not in design] + [design]
    criterion = {}
    for weighted, name in ((True, "event_energy_weighted"), (False, "snapshot_equal")):
        scores = {m: [] for m in data.RANKS}
        for fold in folds:
            keep = np.concatenate([np.arange(a, b) for i, (a, b) in enumerate(zip(ctx["offsets"][:-1], ctx["offsets"][1:])) if i not in fold])
            sub_offsets = np.concatenate([[0], np.cumsum([b - a for i, (a, b) in enumerate(zip(ctx["offsets"][:-1], ctx["offsets"][1:])) if i not in fold])])
            basis = build_basis(ctx["dev"][:, keep], sub_offsets, weighted)[0]
            for m in data.RANKS:
                s = qr_sensors(basis, m)
                d = decoder(basis, s, m)
                scores[m].append(max(float(np.mean([system_nse(blocks_[i], d @ blocks_[i][s]) for i in fold])), -1.0))
        criterion[name] = float(np.mean([np.mean(v) for v in scores.values()]))
    return criterion


def snapshot_equal_dss(ctx: dict) -> dict[int, tuple[list[int], object]]:
    """QR-DSS with the snapshot-equal basis of the 17 development storms (comparison basis)."""
    basis = build_basis(ctx["dev"], ctx["offsets"], weighted=False)[0]
    out = {}
    for m in data.RANKS:
        s = qr_sensors(basis, m)
        d = decoder(basis, s, m)
        out[m] = (s, lambda measured, d=d: d @ measured)
    return out
