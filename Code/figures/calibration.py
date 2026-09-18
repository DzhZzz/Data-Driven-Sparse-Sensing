"""Regenerate manuscript Figure 3 from the frozen calibration event series.

The packaged series are the exact event windows used by the clean one-step
calibration audit. E01 and E04 were calibration events. E02 was excluded from
formal optimization and is reported as an independent hydrologic evaluation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from utils.paths import DATA_DIR, OUTPUTS_DIR

SOURCE_DIR = DATA_DIR
OUTPUT_DIR = OUTPUTS_DIR
SERIES = SOURCE_DIR / "calibration_series.csv"
FIGURE = OUTPUT_DIR / "Figure_3_Calibration_Diagnostic.png"

EXPECTED_METRICS = {
    "E01": {"N": 125, "NSE": 0.5590570574837785, "Observed_Excess_Peak_CMS": 0.0654059993790239,
            "Modeled_Excess_Peak_CMS": 0.07264747813949671, "Peak_Bias_pct": 11.071581856748132,
            "Observed_Excess_Volume_m3": 616.922532382801, "Modeled_Excess_Volume_m3": 540.3632719475755,
            "Volume_Bias_pct": -12.40986613659953},
    "E04": {"N": 205, "NSE": 0.5526996281900551, "Observed_Excess_Peak_CMS": 0.05662611343310331,
            "Modeled_Excess_Peak_CMS": 0.0423441367441507, "Peak_Bias_pct": -25.221537949668726,
            "Observed_Excess_Volume_m3": 549.3282961359595, "Modeled_Excess_Volume_m3": 426.25159254024555,
            "Volume_Bias_pct": -22.404945177856316},
    "E02": {"N": 83, "NSE": 0.6116917385670564, "Observed_Excess_Peak_CMS": 0.28077083557189614,
            "Modeled_Excess_Peak_CMS": 0.1514029448308535, "Peak_Bias_pct": -46.07597170038545,
            "Observed_Excess_Volume_m3": 627.5604280389509, "Modeled_Excess_Volume_m3": 363.08348452190074,
            "Volume_Bias_pct": -42.1436616619547},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nse(observed: np.ndarray, modeled: np.ndarray) -> float:
    denominator = float(np.sum((observed - float(np.mean(observed))) ** 2))
    if denominator <= 0.0:
        raise RuntimeError("NSE denominator is not positive")
    return 1.0 - float(np.sum((modeled - observed) ** 2)) / denominator


def event_metrics(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame["Observed_Excess_CMS"].to_numpy(dtype=float)
    modeled = frame["Modeled_Excess_CMS"].to_numpy(dtype=float)
    timestamps = pd.to_datetime(frame["Timestamp_Local"], errors="raise")
    elapsed_seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    observed_peak = float(np.max(observed))
    modeled_peak = float(np.max(modeled))
    time_steps = np.diff(elapsed_seconds)
    if time_steps.size == 0 or not np.allclose(time_steps, time_steps[0], rtol=0.0, atol=1.0e-9):
        raise RuntimeError("Calibration figure series does not have a fixed time step")
    step_seconds = float(time_steps[0])
    observed_volume = float(np.sum(observed) * step_seconds)
    modeled_volume = float(np.sum(modeled) * step_seconds)
    return {
        "n": int(frame.shape[0]),
        "nse": nse(observed, modeled),
        "observed_peak_m3s": observed_peak,
        "modeled_peak_m3s": modeled_peak,
        "peak_bias_pct": 100.0 * (modeled_peak / observed_peak - 1.0),
        "observed_volume_m3": observed_volume,
        "modeled_volume_m3": modeled_volume,
        "volume_bias_pct": 100.0 * (modeled_volume / observed_volume - 1.0),
    }


def validate(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    required = {
        "Event_ID",
        "Timestamp_Local",
        "Hours_From_Rainfall_Onset",
        "Observed_Excess_CMS",
        "Modeled_Excess_CMS",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Missing calibration-series fields: {sorted(missing)}")
    if set(frame["Event_ID"].unique()) != {"E01", "E04", "E02"}:
        raise RuntimeError("Calibration-series event IDs drifted")

    expected = EXPECTED_METRICS
    calculated: dict[str, dict[str, float]] = {}
    comparisons = {
        "nse": "NSE",
        "observed_peak_m3s": "Observed_Excess_Peak_CMS",
        "modeled_peak_m3s": "Modeled_Excess_Peak_CMS",
        "peak_bias_pct": "Peak_Bias_pct",
        "observed_volume_m3": "Observed_Excess_Volume_m3",
        "modeled_volume_m3": "Modeled_Excess_Volume_m3",
        "volume_bias_pct": "Volume_Bias_pct",
    }
    for event_id in ("E01", "E04", "E02"):
        event = frame.loc[frame["Event_ID"].eq(event_id)].copy()
        metrics = event_metrics(event)
        calculated[event_id] = metrics
        if metrics["n"] != int(expected[event_id]["N"]):
            raise RuntimeError(f"{event_id} row count drifted")
        for actual_name, expected_name in comparisons.items():
            if not np.isclose(
                float(metrics[actual_name]),
                float(expected[event_id][expected_name]),
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise RuntimeError(
                    f"{event_id} {actual_name} drifted: "
                    f"{metrics[actual_name]} != {expected[event_id][expected_name]}"
                )
    return calculated


def make_figure(frame: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            # Match the established manuscript figure hierarchy used by the
            # reconstruction figures: readable panel titles and axis labels,
            # with slightly smaller but still legible tick labels and legend.
            "font.size": 14.0,
            "axes.titlesize": 16.0,
            "axes.labelsize": 16.0,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
            "legend.fontsize": 14.0,
        }
    )
    titles = {
        "E01": "E01 — Calibration",
        "E04": "E04 — Calibration",
        "E02": "E02 — Independent evaluation",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    for axis, event_id in zip(axes, ("E01", "E04", "E02")):
        event = frame.loc[frame["Event_ID"].eq(event_id)]
        axis.plot(
            event["Hours_From_Rainfall_Onset"],
            event["Observed_Excess_CMS"],
            color="#111111",
            linewidth=2.4,
            label="Observed",
        )
        axis.plot(
            event["Hours_From_Rainfall_Onset"],
            event["Modeled_Excess_CMS"],
            color="#0072B2",
            linewidth=2.1,
            label="Frozen clean one-step model",
        )
        axis.set_title(titles[event_id])
        axis.set_xlabel("Time after rainfall onset (h)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Event-excess flow at OF-02 (m$^3$/s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 1.0))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    frame = pd.read_csv(SERIES)
    validate(frame)
    make_figure(frame)
    print(FIGURE)


if __name__ == "__main__":
    main()
