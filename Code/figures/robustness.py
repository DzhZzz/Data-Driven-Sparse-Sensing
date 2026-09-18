"""Regenerate manuscript Figures 8-10 from compact scenario-level inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from scipy.stats import spearmanr

from utils.paths import DATA_DIR, OUTPUTS_DIR, ROOT_DIR


ROOT = ROOT_DIR
DATA = DATA_DIR
OUT = OUTPUTS_DIR
NOISE_CACHE = DATA / "figure8_scenario_values.npz"
MODAL = DATA / "modal_exposure_rows.csv"
SCENARIO_NPZ = DATA / "figure9_signed_loss.npz"
MODAL_METRICS = DATA / "modal_only_metrics.csv"
MODAL_ROBUSTNESS = DATA / "modal_scenario_robustness.csv"
SENSOR_MAP_4 = DATA / "sensor_map_r4.png"
SENSOR_MAP_8 = DATA / "sensor_map_r8.png"
BASIS = DATA / "frozen_basis.npz"
PHYSICAL_DESCRIPTORS = DATA / "node_physical_descriptors.csv"

FIGURE_OUTPUTS = {
    "figure8": {
        "Figure_8_Noise_Effects_Direct_SVD.png",
        "Figure_8_Noise_Summary.csv",
    },
    "figure9": {
        "Figure_9_Single_Sensor_Failure_Direct_SVD.png",
        "Figure_9_Dropout_Summary.csv",
        "Figure_9_r8_sensor_means.csv",
    },
    "figure10": {
        "Figure_10_Modal_Exposure_Failure_Risk_panel_a_scatter_alpha1.png",
        "Figure_10_Modal_Exposure_Scatter_Data.csv",
        "Figure_10_Modal_Only_Metrics.csv",
        "Figure_10_Modal_Scenario_Robustness.csv",
        "Figure_10_Physical_Associations_r8.csv",
        "Figure_10_r8_Node_Physical_Grounding.csv",
        "Figure_10_Interpretive_Model_Metrics.csv",
        "Figure_10_Interpretive_Model_Incremental_R2.csv",
    },
}

FIGURE_INPUTS = {
    "figure8": (NOISE_CACHE,),
    "figure9": (MODAL, SCENARIO_NPZ, SENSOR_MAP_4, SENSOR_MAP_8),
    "figure10": (
        MODAL,
        MODAL_METRICS,
        MODAL_ROBUSTNESS,
        BASIS,
        PHYSICAL_DESCRIPTORS,
    ),
}


PHYSICAL_BOOTSTRAP_SEED = 20260824
INTERPRETIVE_MODEL_BOOTSTRAP_SEED = 20260821


COLORS = {
    "clean": "#1f77b4",
    "5%": "#2ca02c",
    "10%": "#ff7f0e",
    "15%": "#d62728",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_inputs(paths: tuple[Path, ...] | list[Path] | None = None) -> None:
    if paths is None:
        paths = tuple(dict.fromkeys(path for values in FIGURE_INPUTS.values() for path in values))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)


def style_axes(ax: plt.Axes) -> None:
    """Keep the original DSS convention: Arial, framed axes, light grid."""

    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#4a4a4a")
    ax.tick_params(labelsize=10)


def savefig(
    fig: plt.Figure,
    path: Path,
) -> None:
    fig.savefig(
        path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def make_noise(noise_cache: Path) -> dict[str, object]:
    """Plot the 225 scenario-level values retained after MC reduction."""

    with np.load(noise_cache, allow_pickle=False) as payload:
        raw = {
            level: {
                rank: np.asarray(
                    payload[f"{level.replace('%', 'pct')}_r{rank:02d}"], dtype=float
                )
                for rank in range(1, 11)
            }
            for level in ("clean", "5%", "10%", "15%")
        }
    if any(values.shape != (225,) for by_rank in raw.values() for values in by_rank.values()):
        raise AssertionError("Figure 8 requires 225 scenario-level values per box.")

    rows: list[dict[str, object]] = []
    for r in range(1, 11):
        clean = raw["clean"][r]
        if clean.size != 225:
            raise AssertionError(f"clean r={r} has {clean.size} values, expected 225")
        rows.append(
            {
                "scenario": "clean",
                "r": r,
                "summary_unit": "225 held-out scenarios",
                "summary_count": 225,
                "median": float(np.median(clean)),
                "mean": float(np.mean(clean)),
                "q1": float(np.quantile(clean, 0.25)),
                "q3": float(np.quantile(clean, 0.75)),
                "paired_delta_median": 0.0,
                "paired_delta_q1": 0.0,
                "paired_delta_q3": 0.0,
            }
        )
        for level in ("5%", "10%", "15%"):
            scenario_medians = raw[level][r]
            paired_delta = scenario_medians - clean
            rows.append(
                {
                    "scenario": level,
                    "r": r,
                    "summary_unit": "225 scenario medians after MC reduction",
                    "summary_count": 225,
                    "median": float(np.median(scenario_medians)),
                    "mean": float(np.mean(scenario_medians)),
                    "q1": float(np.quantile(scenario_medians, 0.25)),
                    "q3": float(np.quantile(scenario_medians, 0.75)),
                    "paired_delta_median": float(np.median(paired_delta)),
                    "paired_delta_q1": float(np.quantile(paired_delta, 0.25)),
                    "paired_delta_q3": float(np.quantile(paired_delta, 0.75)),
                }
            )
    system = pd.DataFrame(rows).sort_values(["scenario", "r"])
    system.to_csv(OUT / "Figure_8_Noise_Summary.csv", index=False)

    # Reproduce the established DSS visual language: one wide grouped
    # boxplot, with each box retaining 225 scenario-level values.  For noisy
    # levels those values are the MC-major within-scenario medians computed
    # above, not the 225,000 perturbation draws.
    plt.rcParams.update(
        {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]}
    )
    plot_values = raw
    palette = {
        "clean": "#89C0B7",
        "5%": "#B7E1E4",
        "10%": "#6F91B5",
        "15%": "#EF8F88",
    }
    legend_labels = {
        "clean": "Noise-free",
        "5%": "5% Gaussian Noise",
        "10%": "10% Gaussian Noise",
        "15%": "15% Gaussian Noise",
    }
    fig, ax = plt.subplots(figsize=(13.0, 6.8), dpi=300)
    spacing = 4.0
    offsets = (-1.2, -0.4, 0.4, 1.2)
    handles = []
    for level, offset in zip(("clean", "5%", "10%", "15%"), offsets):
        for r in range(1, 11):
            box = ax.boxplot(
                [plot_values[level][r]],
                positions=[r * spacing + offset],
                patch_artist=True,
                widths=0.6,
                flierprops={
                    "marker": "o", "markersize": 2.8,
                    "markerfacecolor": palette[level],
                    "markeredgecolor": palette[level],
                },
                boxprops={"edgecolor": "black", "linewidth": 1.5},
                whiskerprops={"color": "black", "linewidth": 1.4},
                capprops={"color": "black", "linewidth": 1.4},
                medianprops={"color": "black", "linewidth": 1.5},
            )
            box["boxes"][0].set_facecolor(palette[level])
            if r == 1:
                handles.append(box["boxes"][0])
    ax.set_xticks([r * spacing for r in range(1, 11)])
    ax.set_xticklabels([str(r) for r in range(1, 11)])
    ax.set_xlabel("Retained rank, r (= number of sampled nodes)", fontsize=18)
    ax.set_ylabel("System-level NSE", fontsize=18)
    ax.set_ylim(-0.60, 1.01)
    ax.set_yticks(np.arange(-0.6, 1.01, 0.2))
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(axis="y", alpha=0.12, linewidth=0.8)
    ax.legend(
        handles,
        [legend_labels[level] for level in ("clean", "5%", "10%", "15%")],
        fontsize=14,
        loc="lower right",
        frameon=False,
        facecolor="none",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#4a4a4a")
    fig.tight_layout()
    fig.savefig(
        OUT / "Figure_8_Noise_Effects_Direct_SVD.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    r8 = system.loc[system["r"].eq(8)].set_index("scenario")
    return {
        "summary_unit": "225 held-out scenarios; noisy MC reduced within scenario first",
        "r8_medians": {
            key: float(r8.loc[key, "median"])
            for key in ("clean", "5%", "10%", "15%")
        },
        "r8_paired_delta_medians": {
            key: float(r8.loc[key, "paired_delta_median"])
            for key in ("5%", "10%", "15%")
        },
        "r8_15pct_noisy_median": float(r8.loc["15%", "median"]),
        "r8_15pct_paired_delta_median": float(
            r8.loc["15%", "paired_delta_median"]
        ),
        "row_count": int(len(system)),
        "scenario_count": 225,
        "mc_replicates": 1000,
    }


def _draw_source_sensor_panel(ax: plt.Axes, image_path: Path) -> None:
    """Render the established Sensor_4/Sensor_8 canvas without cropping."""

    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image = plt.imread(image_path)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#4a4a4a")
    height, width = image.shape[:2]
    ax.imshow(image, extent=(0, width, height, 0), interpolation="lanczos")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("W")


def make_dropout(modal: pd.DataFrame, matrices: dict[str, np.ndarray]) -> dict[str, object]:
    """Reproduce the original source plotting logic at r=4 and r=8."""

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})
    loss = matrices["loss_matrix"]
    r_values = matrices["r"].astype(int)
    sensors = matrices["sensor_index"].astype(int)
    if loss.shape != (len(modal), 225):
        raise AssertionError(f"unexpected loss matrix shape {loss.shape}")
    if not np.array_equal(r_values, modal["r"].to_numpy(dtype=int)):
        raise AssertionError("modal rows and scenario matrix r ordering differ")
    if not np.array_equal(sensors, modal["Sensor_Index"].to_numpy(dtype=int)):
        raise AssertionError("modal rows and scenario matrix sensor ordering differ")

    rows = []
    for r in range(2, 11):
        idx = np.flatnonzero(r_values == r)
        sensor_mean = loss[idx].mean(axis=1)
        rows.append({
            "r": r,
            "sensor_count": int(len(idx)),
            "mean_sensor_mean_loss": float(sensor_mean.mean()),
            "median_sensor_mean_loss": float(np.median(sensor_mean)),
            "maximum_sensor_mean_loss": float(sensor_mean.max()),
            "minimum_sensor_mean_loss": float(sensor_mean.min()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "Figure_9_Dropout_Summary.csv", index=False)

    display_ranks = (4, 8)
    expected_layouts = {
        4: ["J297", "OF-04", "OF-02", "120"],
        8: ["J297", "OF-03", "J305", "J304", "OF-02", "163", "Stor-01", "94"],
    }
    ordered: dict[int, pd.DataFrame] = {}
    for r in display_ranks:
        idx = np.flatnonzero(r_values == r)
        frame = modal.iloc[idx].copy()
        frame["_row_index"] = idx
        frame["pivot_order"] = frame["pivot_order"].astype(int)
        frame = frame.sort_values("pivot_order")
        actual = frame["Node_ID"].astype(str).tolist()
        if actual != expected_layouts[r]:
            raise AssertionError(f"Current r={r} QR order drifted: {actual}")
        ordered[r] = frame

    r8_idx = np.flatnonzero(r_values == 8)
    r8_rows = modal.iloc[r8_idx].copy()
    r8_rows["sensor_label"] = r8_rows["Node_ID"].astype(str)
    r8_rows["mean_signed_loss"] = loss[r8_idx].mean(axis=1)
    r8_rows["pivot_order"] = r8_rows["pivot_order"].astype(int)
    r8_rows = r8_rows.sort_values("pivot_order")
    r8_top = r8_rows[["Node_ID", "pivot_order", "mean_signed_loss", "Energy_Weighted_Modal_Exposure"]]
    r8_top.to_csv(OUT / "Figure_9_r8_sensor_means.csv", index=False)

    height_ratios = [1.0, 0.08, 1.0, 0.16, 0.82]
    fig = plt.figure(figsize=(10.2, 8.2), dpi=300)
    fig.subplots_adjust(left=0.090, right=0.985, top=0.975, bottom=0.080)
    grid = fig.add_gridspec(
        5, 2, width_ratios=[2.45, 0.78], height_ratios=height_ratios,
        hspace=0.04, wspace=0.00,
    )
    box_axes = []
    sensor_axes = []
    shared_axis = None
    top_delta_values = []
    map_paths = {4: SENSOR_MAP_4, 8: SENSOR_MAP_8}
    for row_index, r in enumerate(display_ranks):
        grid_row = row_index * 2
        ax_box = fig.add_subplot(grid[grid_row, 0], sharey=shared_axis)
        if shared_axis is None:
            shared_axis = ax_box
        ax_sensor = fig.add_subplot(grid[grid_row, 1])
        box_axes.append(ax_box)
        sensor_axes.append(ax_sensor)

        frame = ordered[r]
        row_positions = [int(row["_row_index"]) for _, row in frame.iterrows()]
        box_data = [loss[index] for index in row_positions]
        top_delta_values.extend(value for values in box_data for value in values if np.isfinite(value))
        labels = frame["Node_ID"].astype(str).tolist()
        positions = np.arange(1, len(box_data) + 1)
        ax_box.boxplot(
            box_data, positions=positions, widths=0.58, patch_artist=True,
            boxprops={"facecolor": "#8fb1cc", "edgecolor": "#4a4a4a", "linewidth": 1.5},
            medianprops={"color": "#222222", "linewidth": 1.5},
            whiskerprops={"color": "#4a4a4a", "linewidth": 1.4},
            capprops={"color": "#4a4a4a", "linewidth": 1.4},
            flierprops={"marker": "o", "markersize": 2.6, "markerfacecolor": "#4a98c5", "markeredgecolor": "none", "alpha": 0.45},
        )
        ax_box.axhline(0.0, color="#d65244", linestyle="--", linewidth=1.5, label="No-loss reference")
        ax_box.set_xlim(0.35, len(box_data) + 0.65)
        ax_box.set_xticks(positions)
        ax_box.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
        ax_box.tick_params(axis="y", labelsize=10, pad=2)
        ax_box.grid(True, axis="y", alpha=0.18)
        ax_box.text(0.01, 0.96, f"{chr(ord('a') + row_index)}  r = {r}", transform=ax_box.transAxes,
                    ha="left", va="top", fontsize=12, fontweight="bold")
        if row_index == 0:
            ax_box.legend(loc="upper right", frameon=False, fontsize=10, facecolor="none")
        for spine in ax_box.spines.values():
            spine.set_linewidth(0.9)
            spine.set_edgecolor("#4a4a4a")
        _draw_source_sensor_panel(ax_sensor, map_paths[r])

    if top_delta_values:
        y_min = min(0.0, min(top_delta_values) - 0.03)
        y_max = max(top_delta_values) + 0.06
        for ax in box_axes:
            ax.set_ylim(y_min, y_max)
    box_axes[-1].set_xlabel("Dropped sensor (QR selection order)", fontsize=12, labelpad=2)

    # Align the summary panel with the *visible* outer right frame of the map
    # column.  The map axes use an equal aspect and therefore occupy less than
    # the raw GridSpec cell width; clipping c to the GridSpec cell would leave
    # its right frame visibly beyond the maps.  The map axes themselves are
    # not resized or stretched.  Shrinking the figure's right subplot margin
    # makes the raw grid edge coincide with that fixed map frame.
    fig.canvas.draw()
    map_right = sensor_axes[-1].get_position().x1
    for _ in range(8):
        fig.subplots_adjust(right=map_right)
        fig.canvas.draw()
        updated_map_right = sensor_axes[-1].get_position().x1
        if abs(updated_map_right - map_right) < 1e-7:
            break
        map_right = updated_map_right
    ax_summary = fig.add_subplot(grid[4, :])
    ax_summary.plot(summary["r"], summary["mean_sensor_mean_loss"], marker="o", linestyle="-", linewidth=2,
                    color="#2a7ab0", label="Mean loss")
    ax_summary.plot(summary["r"], summary["median_sensor_mean_loss"], marker="s", linestyle="-", linewidth=2,
                    color="#58a55c", label="Median loss")
    ax_summary.plot(summary["r"], summary["maximum_sensor_mean_loss"], marker="^", linestyle="-", linewidth=2,
                    color="#d65244", label="Worst-case loss")
    ax_summary.set_xlabel("Retained rank, r (= number of sampled nodes)", fontsize=12)
    ax_summary.set_ylim(0.0, 0.8)
    ax_summary.set_xticks(sorted(summary["r"].unique()))
    ax_summary.tick_params(axis="both", labelsize=10, pad=2)
    ax_summary.grid(True, alpha=0.18)
    ax_summary.legend(loc="upper right", frameon=False, fontsize=10, facecolor="none")
    ax_summary.text(0.01, 0.96, "c", transform=ax_summary.transAxes,
                    ha="left", va="top", fontsize=12, fontweight="bold")
    for spine in ax_summary.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#4a4a4a")
    fig.subplots_adjust(left=0.090, right=map_right, top=0.975, bottom=0.080)
    fig.text(0.035, 0.56, "Delta system-level NSE", va="center", rotation="vertical", fontsize=12)
    # One final active-position correction is needed because Matplotlib may
    # re-apply the equal-aspect map boxes during the preceding draw.  Set only
    # c's right edge to that already-rendered map edge; no map geometry changes.
    fig.canvas.draw()
    summary_pos = ax_summary.get_position()
    map_right_final = sensor_axes[-1].get_position().x1
    ax_summary.set_position(
        [summary_pos.x0, summary_pos.y0, map_right_final - summary_pos.x0, summary_pos.height],
        which="both",
    )
    fig.savefig(OUT / "Figure_9_Single_Sensor_Failure_Direct_SVD.png", dpi=300,
                bbox_inches="tight", pad_inches=0.10, facecolor="white")
    plt.close(fig)
    return {
        "scenario_count": int(loss.shape[1]),
        "configuration_count": int(loss.shape[0]),
        "summary": rows,
        "r8_sensor_means": r8_top.to_dict(orient="records"),
        "display": {
            "ranks": list(display_ranks),
            "map_source": [str(SENSOR_MAP_4.relative_to(ROOT)), str(SENSOR_MAP_8.relative_to(ROOT))],
            "node_orders": {str(r): expected_layouts[r] for r in display_ranks},
            "plot_contract": "source dropout combined-delta layout: scenario-level signed-loss boxplots, Sensor_4/Sensor_8 maps, and connected summary across r=2-10",
            "panel_c_ylim": [0.0, 0.8],
            "panel_c_extent": "aligned_to_map_outer_frame",
            "panel_c_summary_max": float(summary[["mean_sensor_mean_loss", "median_sensor_mean_loss", "maximum_sensor_mean_loss"]].to_numpy(dtype=float).max()),
        },
    }


def _normalize_node_id(value: object) -> str:
    return re.sub(r"(OF)-?(\d+)", r"\1-\2", str(value))


def _within_r_zscore(values: np.ndarray, r_values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    r_values = np.asarray(r_values, dtype=int)
    output = np.full_like(values, np.nan, dtype=float)
    for rank in np.unique(r_values):
        mask = (r_values == rank) & np.isfinite(values)
        subset = values[mask]
        if subset.size == 0:
            continue
        scale = float(np.std(subset, ddof=0))
        output[mask] = 0.0 if scale <= 0.0 else (subset - float(np.mean(subset))) / scale
    return output


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.ptp(x[valid]) <= 0.0 or np.ptp(y[valid]) <= 0.0:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def _r2(y_true: np.ndarray, prediction: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    denominator = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if denominator <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((y_true - prediction) ** 2) / denominator)


def _design_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.column_stack(
        [np.ones(len(frame), dtype=float)]
        + [frame[column].to_numpy(dtype=float) for column in columns]
    )


def _fit_ols_predict(
    training: pd.DataFrame,
    testing: pd.DataFrame,
    predictors: list[str],
    response: str,
) -> np.ndarray:
    coefficients = np.linalg.lstsq(
        _design_matrix(training, predictors),
        training[response].to_numpy(dtype=float),
        rcond=None,
    )[0]
    return _design_matrix(testing, predictors) @ coefficients


def _interpretive_model_diagnostic(
    modal: pd.DataFrame,
    descriptors: pd.DataFrame,
    bootstrap_replicates: int = 10000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare prespecified physical and modal explanations without selection.

    The four specifications are an interpretive sensitivity audit, not a
    response-driven model search.  Generalization is evaluated by leaving an
    entire sensor node out; uncertainty resamples nodes rather than treating
    repeated appearances of a node across ranks as independent.
    """

    frame = modal.copy()
    frame["Node_ID"] = frame["Node_ID"].map(_normalize_node_id)
    descriptor_frame = descriptors.copy()
    descriptor_frame["Node_ID"] = descriptor_frame["Node_ID"].map(_normalize_node_id)
    frame = frame.merge(descriptor_frame, on="Node_ID", how="left", validate="many_to_one")
    required = ["Upstream_Area_km2", "Training_Activation_Fraction", "Total_Degree"]
    if frame[required].isna().any().any():
        raise RuntimeError("Physical descriptors are incomplete after node-ID alignment")
    if len(frame) != 3002 or frame["Node_ID"].nunique() != 77:
        raise RuntimeError("Interpretive diagnostic requires 3,002 configurations and 77 nodes")

    ranks = frame["r"].to_numpy(dtype=int)
    frame["Log_Upstream_Area_z"] = _within_r_zscore(
        np.log1p(frame["Upstream_Area_km2"].to_numpy(dtype=float)), ranks
    )
    frame["Training_Activation_Fraction_z"] = _within_r_zscore(
        frame["Training_Activation_Fraction"].to_numpy(dtype=float), ranks
    )
    frame["Log_Total_Degree_z"] = _within_r_zscore(
        np.log1p(frame["Total_Degree"].to_numpy(dtype=float)), ranks
    )
    response = "Delta_Systemwise_Mean_NSE_z"
    specifications = {
        "Upstream area": ["Log_Upstream_Area_z"],
        "Physical descriptors": [
            "Log_Upstream_Area_z",
            "Training_Activation_Fraction_z",
            "Log_Total_Degree_z",
        ],
        "Modal exposure": ["Energy_Weighted_Modal_Exposure_z"],
        "Physical plus modal exposure": [
            "Log_Upstream_Area_z",
            "Training_Activation_Fraction_z",
            "Log_Total_Degree_z",
            "Energy_Weighted_Modal_Exposure_z",
        ],
    }
    nodes = frame["Node_ID"].astype(str).to_numpy()
    predictions: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, object]] = []
    y = frame[response].to_numpy(dtype=float)
    for model, predictors in specifications.items():
        prediction = np.full(len(frame), np.nan, dtype=float)
        for node in np.unique(nodes):
            test_mask = nodes == node
            prediction[test_mask] = _fit_ols_predict(
                frame.loc[~test_mask], frame.loc[test_mask], predictors, response
            )
        if not np.all(np.isfinite(prediction)):
            raise RuntimeError(f"Non-finite leave-one-node-out prediction for {model}")
        predictions[model] = prediction
        fitted = _fit_ols_predict(frame, frame, predictors, response)
        metric_rows.append({
            "model": model,
            "predictors": ",".join(predictors),
            "n_predictors": len(predictors),
            "n_configurations": len(frame),
            "n_unique_nodes": int(frame["Node_ID"].nunique()),
            "in_sample_r2": _r2(y, fitted),
            "leave_one_node_out_r2": _r2(y, prediction),
            "leave_one_node_out_rmse": float(np.sqrt(np.mean(np.square(y - prediction)))),
            "leave_one_node_out_spearman": _safe_spearman(y, prediction),
        })
    metrics = pd.DataFrame(metric_rows)

    unique_nodes = np.asarray(sorted(np.unique(nodes)), dtype=str)
    node_indices = {node: np.flatnonzero(nodes == node) for node in unique_nodes}
    rng = np.random.default_rng(INTERPRETIVE_MODEL_BOOTSTRAP_SEED)
    samples = {model: np.empty(bootstrap_replicates, dtype=float) for model in specifications}
    for iteration in range(bootstrap_replicates):
        sampled_nodes = rng.choice(unique_nodes, size=len(unique_nodes), replace=True)
        sampled_indices = np.concatenate([node_indices[node] for node in sampled_nodes])
        for model in specifications:
            samples[model][iteration] = _r2(y[sampled_indices], predictions[model][sampled_indices])
    comparisons = {
        "Area minus modal exposure": ("Upstream area", "Modal exposure"),
        "Physical minus modal exposure": ("Physical descriptors", "Modal exposure"),
        "Joint minus physical": ("Physical plus modal exposure", "Physical descriptors"),
        "Joint minus modal exposure": ("Physical plus modal exposure", "Modal exposure"),
    }
    summary_rows: list[dict[str, object]] = []
    for label, (left, right) in comparisons.items():
        delta = samples[left] - samples[right]
        summary_rows.append({
            "comparison": label,
            "left_model": left,
            "right_model": right,
            "mean_delta_leave_one_node_out_r2": float(np.mean(delta)),
            "ci025": float(np.quantile(delta, 0.025)),
            "ci975": float(np.quantile(delta, 0.975)),
            "probability_delta_gt_zero": float(np.mean(delta > 0.0)),
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_unit": "sensor node",
        })
    return metrics, pd.DataFrame(summary_rows)


def _physical_grounding_at_r8(
    modal: pd.DataFrame,
    descriptors: pd.DataFrame,
    bootstrap_replicates: int = 10000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relate the current nominal-basis exposure to prespecified descriptors."""

    with np.load(BASIS, allow_pickle=False) as payload:
        basis = np.asarray(payload["nominal_basis"], dtype=float)
        eigenvalues = np.maximum(np.asarray(payload["nominal_eigenvalues"], dtype=float), 0.0)
        node_ids = np.asarray(
            [_normalize_node_id(value) for value in payload["node_ids"]], dtype=str
        )
        weighting = str(np.asarray(payload["nominal_basis_weighting"]).item())
        snapshots = int(np.asarray(payload["nominal_training_snapshot_count"]).item())
    if basis.shape != (77, 77) or eigenvalues.shape != (77,):
        raise RuntimeError("Frozen nominal basis must have shape 77 x 77")
    if weighting != "snapshot_equal_direct_svd" or snapshots != 5824:
        raise RuntimeError("Figure 10 physical grounding requires the current direct-SVD basis")
    trace = np.cumsum(eigenvalues)
    if trace[7] <= 0.0:
        raise RuntimeError("Rank-eight retained energy is non-positive")
    exposure = np.cumsum(np.square(basis) * eigenvalues[None, :], axis=1) / trace[None, :]

    node_frame = descriptors.copy()
    node_frame["Node_ID"] = node_frame["Node_ID"].map(_normalize_node_id)
    node_frame = node_frame.set_index("Node_ID").reindex(node_ids).reset_index()
    if node_frame["Node_ID"].isna().any() or len(node_frame) != 77:
        raise RuntimeError("Physical descriptor table does not align with all 77 basis nodes")
    node_frame["H_r8"] = exposure[:, 7]
    if not np.isclose(float(node_frame["H_r8"].sum()), 1.0, atol=1e-12):
        raise RuntimeError("Rank-eight modal exposures must sum to one")

    selected = modal.loc[modal["r"].eq(8), [
        "Node_ID", "pivot_order", "Energy_Weighted_Modal_Exposure", "Delta_Systemwise_Mean_NSE"
    ]].copy()
    selected["Node_ID"] = selected["Node_ID"].map(_normalize_node_id)
    selected = selected.rename(columns={
        "Energy_Weighted_Modal_Exposure": "selected_H_r8",
        "Delta_Systemwise_Mean_NSE": "selected_mean_signed_loss",
    })
    node_frame = node_frame.merge(selected, on="Node_ID", how="left", validate="one_to_one")
    node_frame["selected_r8_qr"] = node_frame["pivot_order"].notna()
    comparison = node_frame.loc[node_frame["selected_r8_qr"], ["H_r8", "selected_H_r8"]]
    if len(comparison) != 8 or not np.allclose(
        comparison["H_r8"], comparison["selected_H_r8"], atol=1e-12
    ):
        raise RuntimeError("All-node and selected-node rank-eight exposures disagree")

    descriptor_specs = [
        ("Upstream drainage area", "Upstream_Area_km2", True),
        ("Mean circular-conduit diameter", "Mean_Circular_Diameter_m", True),
        ("Node activation fraction", "Training_Activation_Fraction", False),
        ("Total node degree", "Total_Degree", True),
    ]
    association_rows: list[dict[str, object]] = []
    for feature_index, (label, column, log_transform) in enumerate(descriptor_specs):
        values = node_frame[column].to_numpy(dtype=float)
        if log_transform:
            values = np.log1p(values)
        h = node_frame["H_r8"].to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(h)
        x = values[valid]
        y = h[valid]
        rho = _safe_spearman(x, y)
        rng = np.random.default_rng(PHYSICAL_BOOTSTRAP_SEED + feature_index)
        bootstrap = np.empty(bootstrap_replicates, dtype=float)
        for iteration in range(bootstrap_replicates):
            sample = rng.integers(0, len(x), size=len(x))
            bootstrap[iteration] = _safe_spearman(x[sample], y[sample])
        bootstrap = bootstrap[np.isfinite(bootstrap)]
        if len(bootstrap) < bootstrap_replicates * 0.99:
            raise RuntimeError(f"Too many non-finite bootstrap correlations for {label}")
        association_rows.append({
            "descriptor": label,
            "source_column": column,
            "transform_for_association": "log1p" if log_transform else "none",
            "n_nodes": int(valid.sum()),
            "spearman_rho": rho,
            "bootstrap_ci025": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_ci975": float(np.quantile(bootstrap, 0.975)),
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(PHYSICAL_BOOTSTRAP_SEED + feature_index),
        })
    return node_frame, pd.DataFrame(association_rows)


def make_modal(
    modal: pd.DataFrame,
    metrics: pd.DataFrame,
    robustness: pd.DataFrame,
    descriptors: pd.DataFrame,
) -> dict[str, object]:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})
    d = modal.copy()
    d["r"] = d["r"].astype(int)
    d["loss_z_calc"] = d.groupby("r")["Delta_Systemwise_Mean_NSE"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0))
    d["exposure_z_calc"] = d.groupby("r")["Energy_Weighted_Modal_Exposure"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0))
    if not np.allclose(d["loss_z_calc"], d["Delta_Systemwise_Mean_NSE_z"], atol=2e-6, equal_nan=True):
        raise AssertionError("saved loss z-scores disagree with within-r recomputation")
    if not np.allclose(d["exposure_z_calc"], d["Energy_Weighted_Modal_Exposure_z"], atol=2e-6, equal_nan=True):
        raise AssertionError("saved exposure z-scores disagree with within-r recomputation")
    d.to_csv(OUT / "Figure_10_Modal_Exposure_Scatter_Data.csv", index=False)

    # Both variables are standardized independently within rank.  The slope is
    # therefore the modal-only standardized coefficient reported in the paper.
    beta = float(np.dot(d["exposure_z_calc"], d["loss_z_calc"]) /
                 np.dot(d["exposure_z_calc"], d["exposure_z_calc"]))
    if len(metrics) != 1 or len(robustness) != 225:
        raise AssertionError("modal-only inputs must contain one metric row and 225 scenarios")
    metric = metrics.iloc[0]
    if not np.isclose(beta, float(metric["beta_modal_exposure"]), atol=1e-12):
        raise AssertionError("modal coefficient disagrees with the recomputed standardized slope")
    if not np.all(np.isfinite(robustness["spearman"])):
        raise AssertionError("scenario-specific modal correlations contain non-finite values")
    metrics.to_csv(OUT / "Figure_10_Modal_Only_Metrics.csv", index=False)
    robustness.to_csv(OUT / "Figure_10_Modal_Scenario_Robustness.csv", index=False)

    node_frame, physical_associations = _physical_grounding_at_r8(d, descriptors)
    model_metrics, model_increment = _interpretive_model_diagnostic(d, descriptors)
    node_frame.to_csv(OUT / "Figure_10_r8_Node_Physical_Grounding.csv", index=False)
    physical_associations.to_csv(OUT / "Figure_10_Physical_Associations_r8.csv", index=False)
    model_metrics.to_csv(OUT / "Figure_10_Interpretive_Model_Metrics.csv", index=False)
    model_increment.to_csv(
        OUT / "Figure_10_Interpretive_Model_Incremental_R2.csv", index=False
    )

    # Use a compact, manuscript-style arrangement: panel a is the main panel
    # and spans the upper row, while the supporting panels b and c are smaller
    # and share one precisely aligned lower row.
    fig = plt.figure(figsize=(9.5, 7.8), dpi=300)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[0.90, 0.90],
        left=0.14,
        right=0.94,
        top=0.965,
        bottom=0.11,
        hspace=0.08,
        wspace=0.22,
    )
    top_grid = grid[0, :].subgridspec(
        1,
        2,
        width_ratios=[1.0, 0.014],
        wspace=0.085,
    )
    ax = fig.add_subplot(top_grid[0, 0])
    ax_a = ax
    panel_letters: list[object] = []
    ranks = sorted(d["r"].unique())
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(ranks), max(ranks))
    ax.scatter(d["exposure_z_calc"], d["loss_z_calc"], c=d["r"], cmap=cmap, norm=norm,
               s=14, alpha=1.0, linewidths=0, rasterized=True)
    x = np.linspace(float(d["exposure_z_calc"].min()), float(d["exposure_z_calc"].max()), 100)
    ax.plot(x, beta * x, color="#d65244", linestyle="--", linewidth=1.5,
            label="Linear fit")
    ax.set_xlabel(r"Modal exposure (z-score within rank $r$)", fontsize=10)
    ax.set_ylabel(r"Signed dropout loss (z-score within rank $r$)", fontsize=10)
    # Draw each statistic on a fixed baseline grid.  A single multi-line
    # mathtext string gives visibly uneven spacing because superscripts and
    # subscripts have different bounding boxes.
    stats_lines = [
        (rf"$R^2_{{\mathrm{{raw}}}}$ = {float(metric['raw_r2']):.3f}", 0.860),
        (rf"$R^2_{{\mathrm{{LONO}}}}$ = {float(metric['leave_one_node_out_r2']):.3f}", 0.790),
        (rf"$\rho_{{\mathrm{{LONO}}}}$ = {float(metric['leave_one_node_out_spearman']):.3f}", 0.720),
    ]
    for text_line, y_line in stats_lines:
        ax.text(
            0.015,
            y_line,
            text_line,
            transform=ax.transAxes,
            va="baseline",
            ha="left",
            fontsize=9,
        )
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="lower right",
        bbox_to_anchor=(0.97, 0.015),
        borderaxespad=0.4,
    )
    panel_letters.append(ax.annotate(
        "a",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(6, -6),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    ))
    cax = fig.add_subplot(top_grid[0, 1])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(r"Rank $r$", fontsize=9)
    cb.ax.tick_params(labelsize=9)
    style_axes(ax)
    ax.tick_params(labelsize=9)

    ax = fig.add_subplot(grid[1, 0])
    ax_b = ax
    associations = physical_associations.copy()
    # Treat the four descriptors as evenly spaced categorical levels, but
    # leave a small, symmetric inset from the panel borders so the end levels
    # do not sit directly on the frame.
    positions = np.linspace(0.18, 2.82, len(associations))[::-1]
    rho = associations["spearman_rho"].to_numpy(dtype=float)
    lower = associations["bootstrap_ci025"].to_numpy(dtype=float)
    upper = associations["bootstrap_ci975"].to_numpy(dtype=float)
    labels = [
        "Drainage area",
        "Conduit diameter",
        "Node activation\nfraction",
        "Node degree",
    ]
    colors = ["#5b8db8", "#6fae9f", "#b184be", "#c99b32"]
    markers = ["o", "^", "D", "x"]
    for position, lo, hi, color in zip(positions, lower, upper, colors):
        ax.plot([lo, hi], [position, position], color=color, alpha=0.18,
                linewidth=11.0, solid_capstyle="round", zorder=1)
    ax.errorbar(
        rho,
        positions,
        xerr=np.vstack([rho - lower, upper - rho]),
        fmt="none",
        ecolor="#4a4a4a",
        elinewidth=1.35,
        capsize=4,
        zorder=2,
    )
    for position, value, color, marker in zip(positions, rho, colors, markers):
        if marker == "x":
            ax.scatter(value, position, s=58, marker=marker, color=color,
                       linewidth=1.1, zorder=3)
        else:
            ax.scatter(value, position, s=58, marker=marker, color=color,
                       edgecolor="white", linewidth=0.7, zorder=3)
    ax.axvline(0.0, color="#d65244", linestyle="--", linewidth=1.2)
    ax.set_yticks(positions, labels, fontsize=9)
    ax.set_ylim(-0.15, 3.15)
    for tick_label in ax.get_yticklabels():
        tick_label.set_rotation(0)
        tick_label.set_rotation_mode("anchor")
        tick_label.set_ha("right")
        tick_label.set_va("center")
    ax.set_xlim(-0.25, 1.05)
    ax.set_xlabel(r"Spearman $\rho$ across nodes", fontsize=10)
    panel_letters.append(ax.annotate(
        "b",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(6, -6),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    ))
    style_axes(ax)
    ax.tick_params(axis="x", labelsize=9, pad=2)
    ax.tick_params(axis="y", labelsize=9, pad=2)
    ax.set_box_aspect(0.82)

    ax = fig.add_subplot(grid[1, 1])
    ax_c = ax
    ax.scatter(
        node_frame["Upstream_Area_km2"],
        node_frame["H_r8"],
        s=30,
        color="#8fbcd4",
        edgecolor="white",
        linewidth=0.55,
        alpha=0.88,
        label="All candidate nodes",
        zorder=2,
    )
    selected_nodes = node_frame.loc[node_frame["selected_r8_qr"]].copy()
    ax.scatter(
        selected_nodes["Upstream_Area_km2"],
        selected_nodes["H_r8"],
        s=66,
        marker="o",
        color="#d65244",
        edgecolor="white",
        linewidth=0.9,
        label=r"$r = 8$ QR sensors",
        zorder=4,
    )
    # Use a common annotation distance, text size, and weight.  Mirrored
    # offsets keep the labels clear of their points while making the leader
    # lines visually comparable across all three annotations.
    label_offsets = {"J304": (12, -12), "OF-02": (12, -12), "94": (-12, 12)}
    annotation_fontsize = 8.5
    annotation_fontweight = "bold"
    annotation_specs = []
    for node, offset in label_offsets.items():
        row = selected_nodes.loc[selected_nodes["Node_ID"].eq(node)]
        if len(row) != 1:
            raise RuntimeError(f"Expected one r=8 selected node for annotation: {node}")
        item = row.iloc[0]
        annotation_specs.append(
            (
                node,
                float(item["Upstream_Area_km2"]),
                float(item["H_r8"]),
                offset,
            )
        )
    area_row = associations.loc[associations["descriptor"].eq("Upstream drainage area")].iloc[0]
    c_stats_text = ax.text(
        0.015,
        0.890,
        fr"$\rho$ = {float(area_row['spearman_rho']):.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    ax.set_xlabel("Upstream drainage area (km$^2$)", fontsize=10)
    ax.set_ylabel(r"Rank-8 modal exposure $H_{i,8}$", fontsize=10)
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="lower right",
        handlelength=0.8,
        handletextpad=0.50,
    )
    panel_letters.append(ax.annotate(
        "c",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(6, -6),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    ))
    style_axes(ax)
    ax.tick_params(labelsize=9, pad=2)
    ax.set_box_aspect(0.82)

    # Use a lighter, consistent horizontal grid across all three panels.
    for panel_ax in (ax_a, ax_b, ax_c):
        panel_ax.grid(axis="y", color="#e6e6e6", linewidth=0.45, alpha=0.55)

    # Keep the narrow colorbar next to panel a, with a deliberate visual gap
    # from the panel frame, and match its height to the actual a-panel box.
    fig.canvas.draw()
    a_box = ax_a.get_position().frozen()
    colorbar_gap = 0.009
    colorbar_width = 0.009
    cax.set_position(
        [a_box.x1 + colorbar_gap, a_box.y0, colorbar_width, a_box.height]
    )

    # The two lower panels are intentionally the same size and baseline.  Set
    # their final positions explicitly so long tick labels cannot introduce a
    # small vertical mismatch during rendering.  Their outer frame edges are
    # anchored to panel a: b shares a's left edge and c shares a's right edge.
    b_box = ax_b.get_position().frozen()
    c_box = ax_c.get_position().frozen()
    common_y0 = min(b_box.y0, c_box.y0)
    common_height = min(b_box.height, c_box.height)
    # Leave a wider central gutter for the long y-axis title of panel c.
    lower_width = min(b_box.width, c_box.width) * 0.94
    ax_b.set_position([a_box.x0, common_y0, lower_width, common_height])
    ax_c.set_position([a_box.x1 - lower_width, common_y0, lower_width, common_height])

    # Draw the annotations only after the final panel positions are fixed.
    # This keeps the display-space leader-line lengths identical in the saved
    # figure rather than letting later axes resizing stretch them differently.
    fig.canvas.draw()
    # Equalize the *rendered* panel-letter insets, not only their anchor points.
    # Lowercase glyphs have different ascender/side-bearing geometry (notably
    # ``b`` versus ``a``/``c``), which otherwise creates a visible misalignment.
    renderer = fig.canvas.get_renderer()
    panel_axes = (ax_a, ax_b, ax_c)
    target_inset_px = 6.0 * fig.dpi / 72.0
    # Compensate for the different ink-bearing top of the bold lowercase
    # glyphs so that the visible (not merely typographic) top edges align.
    glyph_top_correction_pt = {"a": 0.0, "b": 1.8, "c": 0.0}
    for panel_ax, panel_letter in zip(panel_axes, panel_letters):
        frame = panel_ax.get_window_extent(renderer)
        rendered = panel_letter.get_window_extent(renderer)
        dx_px = (frame.x0 + target_inset_px) - rendered.x0
        correction_px = glyph_top_correction_pt[panel_letter.get_text()] * fig.dpi / 72.0
        dy_px = (frame.y1 - target_inset_px - correction_px) - rendered.y1
        current_offset = panel_letter.get_position()
        panel_letter.set_position(
            (
                current_offset[0] + dx_px * 72.0 / fig.dpi,
                current_offset[1] + dy_px * 72.0 / fig.dpi,
            )
        )
    fig.canvas.draw()
    # Align panel c's correlation annotation with the visible left edge of
    # the panel letter, keeping the annotation inside the same text column.
    c_frame = ax_c.get_window_extent(renderer)
    c_letter_box = panel_letters[2].get_window_extent(renderer)
    c_stats_box = c_stats_text.get_window_extent(renderer)
    c_stats_shift_axes = (c_letter_box.x0 - c_stats_box.x0) / c_frame.width
    c_stats_position = c_stats_text.get_position()
    c_stats_text.set_position(
        (c_stats_position[0] + c_stats_shift_axes, c_stats_position[1])
    )
    fig.canvas.draw()

    # Align the visible top edges of the two lower x-axis titles.  Mathtext
    # glyphs (rho versus the superscript in km^2) have different ink-bearing
    # extents even when Matplotlib gives them the same automatic label pad.
    lower_title_pairs = (
        (ax_b, ax_b.xaxis.label),
        (ax_c, ax_c.xaxis.label),
    )
    title_boxes = [
        title.get_window_extent(renderer)
        for _, title in lower_title_pairs
    ]
    target_title_top = min(box.y1 for box in title_boxes)
    # The mathtext in the c-title has a lower visible ink edge than the
    # corresponding rho title in b, despite identical typographic extents.
    # A small renderer-specific correction keeps the visible title tops level.
    visible_top_correction_pt = {ax_b: 0.0, ax_c: 1.8}
    for (panel_ax, title), box in zip(lower_title_pairs, title_boxes):
        frame = panel_ax.get_window_extent(renderer)
        current_position = title.get_position()
        current_y_axes = panel_ax.transAxes.inverted().transform(
            (0.0, current_position[1])
        )[1]
        title_shift_axes = (
            (target_title_top - box.y1)
            + visible_top_correction_pt[panel_ax] * fig.dpi / 72.0
        ) / frame.height
        panel_ax.xaxis.set_label_coords(
            0.5,
            current_y_axes + title_shift_axes,
            transform=panel_ax.transAxes,
        )
    fig.canvas.draw()

    for node, point_x, point_y, offset in annotation_specs:
        point_xy = np.asarray(ax_c.transData.transform((point_x, point_y)), dtype=float)
        offset_px = np.asarray(offset, dtype=float) * fig.dpi / 72.0
        label_xy = ax_c.transData.inverted().transform(point_xy + offset_px)
        ax_c.plot(
            [point_x, float(label_xy[0])],
            [point_y, float(label_xy[1])],
            color="#666666",
            linewidth=0.7,
            solid_capstyle="round",
            zorder=4,
        )
        ax_c.text(
            float(label_xy[0]),
            float(label_xy[1]),
            node,
            ha="left" if offset[0] > 0 else "right",
            va="top" if offset[1] < 0 else "bottom",
            fontsize=annotation_fontsize,
            fontweight=annotation_fontweight,
            color="#3a3a3a",
            zorder=5,
        )

    overall_median = float(metric["scenario_spearman_median"])
    savefig(fig, OUT / "Figure_10_Modal_Exposure_Failure_Risk_panel_a_scatter_alpha1.png")
    metric_lookup = model_metrics.set_index("model")
    increment_lookup = model_increment.set_index("comparison")
    return {
        "row_count": int(len(d)),
        "rank_count": int(d["r"].nunique()),
        "beta_modal_exposure": beta,
        "raw_r2": float(metric["raw_r2"]),
        "leave_one_node_out_r2": float(metric["leave_one_node_out_r2"]),
        "leave_one_node_out_spearman": float(metric["leave_one_node_out_spearman"]),
        "scenario_spearman_median": overall_median,
        "scenario_spearman_q05": float(metric["scenario_spearman_q05"]),
        "scenario_spearman_q95": float(metric["scenario_spearman_q95"]),
        "scenario_spearman_positive_fraction": float(metric["scenario_spearman_positive_fraction"]),
        "physical_grounding_rank": 8,
        "physical_associations": {
            str(row.descriptor): {
                "n_nodes": int(row.n_nodes),
                "spearman_rho": float(row.spearman_rho),
                "ci025": float(row.bootstrap_ci025),
                "ci975": float(row.bootstrap_ci975),
            }
            for row in physical_associations.itertuples(index=False)
        },
        "interpretive_model_leave_one_node_out_r2": {
            str(model): float(value)
            for model, value in metric_lookup["leave_one_node_out_r2"].items()
        },
        "joint_minus_modal_bootstrap": {
            "mean_delta": float(increment_lookup.loc[
                "Joint minus modal exposure", "mean_delta_leave_one_node_out_r2"
            ]),
            "ci025": float(increment_lookup.loc["Joint minus modal exposure", "ci025"]),
            "ci975": float(increment_lookup.loc["Joint minus modal exposure", "ci975"]),
        },
        "physical_interpretation_scope": (
            "prespecified descriptive grounding and incremental diagnostic; "
            "no response-driven selection and no causal interpretation"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate one manuscript robustness figure or all three."
    )
    parser.add_argument(
        "--figure",
        choices=("figure8", "figure9", "figure10", "all"),
        default="figure10",
        help="figure action to run; the staged default is figure10",
    )
    args = parser.parse_args()
    requested_figures = ("figure8", "figure9", "figure10") if args.figure == "all" else (args.figure,)
    source_paths = sorted(
        {path for figure in requested_figures for path in FIGURE_INPUTS[figure]},
        key=lambda path: str(path),
    )
    require_inputs(source_paths)
    OUT.mkdir(parents=True, exist_ok=True)
    modal = pd.read_csv(MODAL) if {"figure9", "figure10"}.intersection(requested_figures) else None
    matrices = None
    if "figure9" in requested_figures:
        with np.load(SCENARIO_NPZ, allow_pickle=False) as matrices_npz:
            matrices = {k: matrices_npz[k] for k in matrices_npz.files}
    metrics = pd.read_csv(MODAL_METRICS) if "figure10" in requested_figures else None
    robustness = pd.read_csv(MODAL_ROBUSTNESS) if "figure10" in requested_figures else None
    descriptors = pd.read_csv(PHYSICAL_DESCRIPTORS) if "figure10" in requested_figures else None
    noise_summary = None
    dropout_summary = None
    modal_summary = None
    if args.figure in {"figure8", "all"}:
        noise_summary = make_noise(NOISE_CACHE)
    if args.figure in {"figure9", "all"}:
        dropout_summary = make_dropout(modal, matrices)
    if args.figure in {"figure10", "all"}:
        modal_summary = make_modal(modal, metrics, robustness, descriptors)
    selected_outputs = set().union(
        *(FIGURE_OUTPUTS[key] for key in FIGURE_OUTPUTS if args.figure in {key, "all"})
    )
    output_paths = sorted(
        OUT / name for name in selected_outputs if (OUT / name).is_file()
    )
    print(json.dumps({"output_dir": str(OUT), "figures": [p.name for p in output_paths]}, indent=2))


if __name__ == "__main__":
    main()
