"""Regenerate the manuscript-specific reconstruction figures (Figures 4--6).

The calculations consume only the frozen snapshot-equal direct-SVD basis,
the frozen QR layouts, and the 225 held-out simulations distributed with the
submission package.  Plotting deliberately reuses the legacy visual grammar
already used by the manuscript: Arial typography, the blue/red event palette,
half violins on a signed-log NSE axis, and the original shadow-line colors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Patch
from matplotlib.ticker import (
    FuncFormatter,
    FormatStrFormatter,
    MaxNLocator,
    PercentFormatter,
)

from utils.paths import DATA_DIR, OUTPUTS_DIR, ROOT_DIR
from utils.plotting import (
    REAL_COLOR,
    DESIGN_COLOR,
    draw_half_violin,
    set_negative_log_axis,
    plot_system_combined,
)
from utils.reconstruction import (
    calculate_metrics,
    load_context,
    projection_operators,
    reconstruct,
)


ROOT = ROOT_DIR
RESULTS_ROOT = OUTPUTS_DIR
NODE_LEVEL_DIR = RESULTS_ROOT
SYSTEM_LEVEL_DIR = RESULTS_ROOT
HELD_OUT_HYDROGRAPH_DIR = RESULTS_ROOT
FIGURE4_NAME = "Figure_4_System_Level_Reconstruction_Direct_SVD.png"
FIGURE5_NAME = "Figure_5_Heldout_Hydrographs_Direct_SVD.png"
FIGURE6_NAME = "Figure_6_Node_Level_Typical_Performance_and_Local_Failure_Risk.png"
FIGURE5_RANKS = (1, 8, 10)
F4_SCATTER_RANKS = (1, 8, 10)
FIGURE5_TEXT_SIZE = 16


def _style_axis(ax: plt.Axes, tick_fontsize: float = 12.5) -> None:
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#4a4a4a")


def _format_elapsed_time(value: float, _position: float) -> str:
    """Show elapsed-hour ticks in the original HH:MM display convention."""
    total_minutes = int(round(float(value) * 60.0))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:02d}:{minutes:02d}"


def _format_flow_tick(value: float, _position: float) -> str:
    """Use two decimals and suppress any negative auto-margin labels."""
    value = float(value)
    return "" if value < -1.0e-10 else f"{value:.2f}"


def generate_figure4(context: dict[str, Any], output_dir: Path = SYSTEM_LEVEL_DIR) -> dict[str, Any]:
    """Generate the system-level rank profile and three reconstruction panels."""

    # Set the manuscript font contract before Figure 4 is constructed. Without
    # this local initialization, Figure 4 depended on whether another plotting
    # function had already mutated Matplotlib's process-global rcParams.
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})
    ranks = tuple(range(1, 11))
    metrics = calculate_metrics(context, ranks)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / FIGURE4_NAME
    plot_system_combined(metrics, context, ranks, figure_path, "line", scatter_ranks=F4_SCATTER_RANKS)

    rows: list[dict[str, Any]] = []
    for event_type in ("observed", "design"):
        for rank in ranks:
            values = np.asarray(metrics["system"][event_type][rank], dtype=float)
            rows.append({
                "Event_Type": event_type,
                "Sensor_Count": rank,
                "Scenario_Count": int(np.isfinite(values).sum()),
                "Mean_NSE": float(np.nanmean(values)),
                "Minimum_NSE": float(np.nanmin(values)),
                "Maximum_NSE": float(np.nanmax(values)),
            })
    summary_path = output_dir / "Figure_4_System_Level_Summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    panel_contract = output_dir / "Figure_4_Panel_Contract.json"
    panel_contract.write_text(json.dumps({
        "scatter_ranks": list(F4_SCATTER_RANKS),
        "top_profile_ranks": list(ranks),
        "data_scope": "225 held-out node_total_inflow simulations: 200 observed-event and 25 200-year design-event cases",
        "state_variable": "node_total_inflow",
        "scatter_tick_format": "one_decimal",
        "font_contract": {
            "panel_labels_pt": 16,
            "axis_labels_pt": 16,
            "tick_labels_pt": 14,
            "legend_labels_pt": 14,
        },
        "y_label_alignment": "top_and_lower_rows_aligned",
        "outer_margin_contract": "Figure_9_style_tight_padding",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"figure": str(figure_path), "summary": str(summary_path), "ranks": list(ranks), "scatter_ranks": list(F4_SCATTER_RANKS), "panel_contract": str(panel_contract)}


def _select_shadowline_events(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Select three performance-independent hydrologic cases for Figure 5.

    The candidate pool is restricted to the eight observed 2025 held-out
    rainfall events.  The three deterministic roles are the minimum event
    depth, the maximum parameter-ensemble mean network-response peak, and the
    maximum event depth.  Reconstruction errors, NSE, and DSS layout
    performance are not used for selection.
    """

    catalog = pd.read_csv(DATA_DIR / "rainfall_catalog.csv", low_memory=False)
    catalog = catalog.loc[
        catalog["Split"].astype(str).str.lower().eq("testing")
        & catalog["Event_Type"].astype(str).str.lower().eq("observed")
    ].copy()
    catalog["Rain_Total_mm"] = pd.to_numeric(catalog["Rain_Total_mm"], errors="coerce")
    catalog = catalog.dropna(subset=["Rain_Total_mm"]).sort_values(
        "Rain_Total_mm", kind="mergesort"
    ).reset_index(drop=True)
    observed_events = [
        event for event in context["events"]
        if str(event.get("event_type", "")).lower() == "observed"
    ]
    events_by_catalog: dict[str, list[dict[str, Any]]] = {}
    for event in observed_events:
        events_by_catalog.setdefault(str(event.get("catalog_id")), []).append(event)
    for catalog_id, events in events_by_catalog.items():
        if len(events) != 25:
            raise RuntimeError(
                f"Figure 5 requires 25 parameter alternatives for {catalog_id}; got {len(events)}"
            )

    def ensemble_network_peak(row: pd.Series) -> float:
        events = events_by_catalog[str(row["Catalog_ID"])]
        values = np.stack([np.asarray(event["data"], dtype=float) for event in events], axis=0)
        # Average the parameter alternatives at each node and time before
        # taking the network mean; this preserves the original node-wise
        # shadow interval in the plotting function below.
        return float(np.max(np.mean(values, axis=(0, 1))))

    catalog["Ensemble_Network_Peak_m3s"] = catalog.apply(ensemble_network_peak, axis=1)
    totals = catalog["Rain_Total_mm"].to_numpy(dtype=float)
    minimum_depth_row = catalog.iloc[int(np.argmin(totals))]
    peak_dominant_row = catalog.iloc[
        int(np.argmax(catalog["Ensemble_Network_Peak_m3s"].to_numpy(dtype=float)))
    ]
    maximum_depth_row = catalog.iloc[int(np.argmax(totals))]
    selected_rows = (minimum_depth_row, peak_dominant_row, maximum_depth_row)
    selected_ids = [str(row["Catalog_ID"]) for row in selected_rows]
    if len(set(selected_ids)) != 3:
        raise RuntimeError(
            "The minimum-depth, maximum-peak, and maximum-depth criteria did not "
            f"produce three distinct events: {selected_ids}"
        )

    selected: list[dict[str, Any]] = []
    roles = ("Minimum depth", "Peak dominant", "Maximum depth")
    labels = ("Minimum-depth event", "Peak-dominant event", "Maximum-depth event")
    for role, label, row in zip(roles, labels, selected_rows):
        events = sorted(
            events_by_catalog[str(row["Catalog_ID"])],
            key=lambda event: str(event.get("parameter_set_id", "")),
        )
        rain_date = str(row.get("Rain_Start_Local", "")).split(" ")[0]
        selected.append({
            "size": role,
            "catalog_id": str(row["Catalog_ID"]),
            "events": events,
            "parameter_count": len(events),
            "rain_total_mm": float(row["Rain_Total_mm"]),
            "max_intensity_mm_h": float(row.get("Max_Equivalent_Intensity_mm_h", np.nan)),
            "ensemble_network_peak_m3s": float(row["Ensemble_Network_Peak_m3s"]),
            "display_label": f"{label} | {rain_date} | total={float(row['Rain_Total_mm']):.2f} mm",
        })
    expected_totals = np.asarray([10.37, 26.14, 41.20])
    actual_totals = np.asarray([item["rain_total_mm"] for item in selected])
    if not np.allclose(actual_totals, expected_totals, atol=0.005):
        raise AssertionError(f"Figure 5 rainfall selection drifted: {actual_totals.tolist()}")

    selected_role_by_id = {item["catalog_id"]: item["size"] for item in selected}
    audit = catalog.copy()
    start = pd.to_datetime(audit["Rain_Start_Local"], errors="coerce")
    end = pd.to_datetime(audit["Rain_End_Local"], errors="coerce")
    audit["Duration_h"] = (end - start).dt.total_seconds() / 3600.0
    audit["Figure_5_Role"] = audit["Catalog_ID"].map(selected_role_by_id).fillna("Not selected")
    audit[
        [
            "Catalog_ID",
            "Rain_Start_Local",
            "Duration_h",
            "Rain_Total_mm",
            "Max_Equivalent_Intensity_mm_h",
            "Ensemble_Network_Peak_m3s",
            "Figure_5_Role",
        ]
    ].to_csv(HELD_OUT_HYDROGRAPH_DIR / "Figure_5_Candidate_Event_Audit.csv", index=False)
    return selected


def _time_axis(labels: list[str], length: int) -> np.ndarray:
    parsed = pd.to_datetime(list(labels[:length]), errors="coerce")
    if len(parsed) == length and not parsed.isna().any():
        return (parsed - parsed[0]).total_seconds().to_numpy(dtype=float) / 3600.0
    return np.arange(length, dtype=float) * (5.0 / 60.0)


def _add_error_inset(
    ax: plt.Axes,
    time_axis: np.ndarray,
    observed_mean: np.ndarray,
    reconstructed_means: dict[int, np.ndarray],
) -> None:
    observed_peak = float(np.nanmax(observed_mean))
    observed_peak_index = int(np.nanargmax(observed_mean))
    observed_integral = float(np.trapezoid(observed_mean, time_axis))
    rank_errors: dict[int, np.ndarray] = {}
    for rank in FIGURE5_RANKS:
        values = reconstructed_means[rank]
        peak_index = int(np.nanargmax(values))
        rank_errors[rank] = np.asarray([
            abs(float(np.nanmax(values)) - observed_peak) / observed_peak * 100.0 if observed_peak else 0.0,
            float(np.round(
                abs(float(time_axis[peak_index]) - float(time_axis[observed_peak_index])) * 60.0,
                10,
            )),
            abs(float(np.trapezoid(values, time_axis)) - observed_integral) / observed_integral * 100.0
            if observed_integral else 0.0,
        ], dtype=float)

    def format_percent(value: float) -> str:
        if value >= 100.0:
            return f"{value:.1f}"
        if value >= 0.1:
            return f"{value:.2f}"
        return f"{value:.3f}"

    cell_text = [
        [r"$E_{Q_{\mathrm{p}}}$ (%)"]
        + [format_percent(rank_errors[rank][0]) for rank in FIGURE5_RANKS],
        [r"$|\Delta T_{\mathrm{p}}|$ (min)"]
        + [f"{rank_errors[rank][1]:.0f}" for rank in FIGURE5_RANKS],
        [r"$E_V$ (%)"]
        + [format_percent(rank_errors[rank][2]) for rank in FIGURE5_RANKS],
    ]
    inset = ax.inset_axes([0.050, 0.625, 0.305, 0.235])
    inset.axis("off")
    table = inset.table(
        cellText=cell_text,
        colLabels=["", r"$r=1$", r"$r=8$", r"$r=10$"],
        cellLoc="center",
        rowLoc="center",
        colLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
        colWidths=[0.35, 0.2167, 0.2167, 0.2166],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(FIGURE5_TEXT_SIZE - 4)
    table.scale(1.0, 1.05)
    rank_colors = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}
    for (row, column), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("#b8b8b8")
        cell.set_linewidth(0.65)
        cell.PAD = 0.035
        if row == 0 and column in rank_colors:
            cell.get_text().set_color(rank_colors[column])
            cell.get_text().set_weight("bold")
        if column == 0 and row > 0:
            cell.get_text().set_ha("left")


def _plot_shadow_panel(ax: plt.Axes, panel: dict[str, Any], tag: str) -> dict[str, Any]:
    palette = {
        1: ("#1f77b4", "#c6dbef", "s"),
        8: ("#2ca02c", "#c7e9c0", "^"),
        10: ("#d62728", "#fcbba1", "D"),
    }
    observed = np.asarray(panel["observed"], dtype=float).T
    time_axis = _time_axis(panel["time_labels"], observed.shape[0])
    observed_mean = np.mean(observed, axis=1)
    ax.plot(time_axis, observed_mean, linewidth=2.4, color="#6a51a3", marker="o",
            markersize=6.6, markevery=max(1, len(time_axis) // 12), label="SWMM-simulated mean", zorder=4)
    ax.fill_between(time_axis, np.nanpercentile(observed, 10, axis=1),
                    np.nanpercentile(observed, 90, axis=1), color="#dadaeb", alpha=0.32,
                    edgecolor="none", linewidth=0, zorder=4)
    reconstructed_means: dict[int, np.ndarray] = {}
    for rank, zorder in ((10, 2), (8, 5), (1, 6)):
        values = np.asarray(panel["reconstructed"][rank], dtype=float).T
        mean = np.mean(values, axis=1)
        reconstructed_means[rank] = mean
        line, fill, marker = palette[rank]
        ax.plot(time_axis, mean, linewidth=2.0, color=line, marker=marker,
                markersize=6.2, markevery=max(1, len(time_axis) // 12),
                label=f"r={rank} mean", zorder=zorder)
        ax.fill_between(time_axis, np.nanpercentile(values, 10, axis=1),
                        np.nanpercentile(values, 90, axis=1), color=fill, alpha=0.26,
                        edgecolor="none", linewidth=0, zorder=max(1, zorder - 1))
    _add_error_inset(ax, time_axis, observed_mean, reconstructed_means)
    ax.add_artist(AnchoredText(tag, loc="upper left", prop={"size": FIGURE5_TEXT_SIZE, "weight": "bold"}, frameon=False))
    ax.add_artist(AnchoredText(panel["display_label"], loc="upper right", prop={"size": FIGURE5_TEXT_SIZE}, frameon=False))
    # Keep the short event legible without changing the longer-event cadence.
    # The small event spans only about 14 h, so a 2-h locator avoids the sparse
    # 0/4/8/12 pattern while the medium and large events retain 4-h spacing.
    tick_step = 2.0 if float(time_axis[-1]) <= 16.0 else 4.0
    ticks = np.arange(0.0, max(tick_step, float(time_axis[-1])) + 1.0e-9, tick_step)
    # The origin is implicit from the event-window boundary; omit its label
    # and tick mark to match the manuscript's original presentation.
    ticks = ticks[ticks > 0.0]
    if str(tag).lower() == "a" and ticks.size:
        # The short panel's terminal tick crowds the right-hand event label;
        # retain the corresponding axis limit but omit that final tick.
        ticks = ticks[:-1]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_elapsed_time))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_flow_tick))
    # Use the same number of major y ticks in all three panels so the visual
    # density is comparable even though the event magnitudes differ.
    # Preserve Matplotlib's automatic lower margin, but prune any negative
    # major tick so the displayed flow scale starts at the original 0 label.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=7, prune="lower"))
    ax.set_xlim(0.0, float(time_axis[-1]))
    ax.tick_params(axis="both", labelsize=FIGURE5_TEXT_SIZE)
    observed_peak_index = int(np.nanargmax(observed_mean))
    return {
        str(rank): {
            "peak_error_percent": abs(float(np.nanmax(reconstructed_means[rank])) - float(np.nanmax(observed_mean)))
            / float(np.nanmax(observed_mean)) * 100.0,
            "absolute_peak_time_difference_minutes": float(np.round(
                abs(
                    float(time_axis[int(np.nanargmax(reconstructed_means[rank]))])
                    - float(time_axis[observed_peak_index])
                ) * 60.0,
                10,
            )),
            "event_volume_error_percent": abs(float(np.trapezoid(reconstructed_means[rank], time_axis)) - float(np.trapezoid(observed_mean, time_axis)))
            / float(np.trapezoid(observed_mean, time_axis)) * 100.0,
        }
        for rank in FIGURE5_RANKS
    }


def generate_figure5(context: dict[str, Any], output_dir: Path = HELD_OUT_HYDROGRAPH_DIR) -> dict[str, Any]:
    selected = _select_shadowline_events(context)
    operators = projection_operators(context["basis"], context["sensor_sets"])
    panels: list[dict[str, Any]] = []
    for item in selected:
        event_matrices = [np.asarray(event["data"], dtype=float) for event in item["events"]]
        # Preserve node-wise spread for the shadow interval: average the 25
        # parameter alternatives at each node and time, then let the original
        # panel routine take the 10th--90th percentile across the 77 nodes.
        observed = np.mean(np.stack(event_matrices, axis=0), axis=0)
        panels.append({
            "observed": observed,
            "reconstructed": {
                rank: np.mean(
                    np.stack([
                        reconstruct(
                            matrix,
                            operators[rank],
                            context["sensor_sets"][rank],
                            context["scale"],
                        )
                        for matrix in event_matrices
                    ], axis=0),
                    axis=0,
                )
                for rank in FIGURE5_RANKS
            },
            "time_labels": item["events"][0].get("time_labels", []),
            "display_label": item["display_label"],
        })
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), dpi=300, sharex=False)
    metrics: list[dict[str, Any]] = []
    for ax, panel, tag, item in zip(axes, panels, ("a", "b", "c"), selected):
        panel_metrics = _plot_shadow_panel(ax, panel, tag)
        metrics.append({"size": item["size"], "scenario_id": item["events"][0]["event_name"],
                        "catalog_id": item["catalog_id"],
                        "parameter_count": item["parameter_count"],
                        "parameter_set_ids": [
                            str(event.get("parameter_set_id", "")) for event in item["events"]
                        ],
                        "rain_total_mm": item["rain_total_mm"], "errors": panel_metrics})
    fig.text(0.015, 0.5, "Parameter-ensemble mean of network-mean nodal inflow (m³/s)", va="center",
             rotation="vertical", fontsize=FIGURE5_TEXT_SIZE)
    # Attach the shared x-axis title to the bottom axes so its distance from
    # the tick labels follows the same axis-label spacing as the y-axis title.
    axes[-1].set_xlabel("Elapsed time (h)", fontsize=FIGURE5_TEXT_SIZE, labelpad=6)
    handles, labels = axes[0].get_legend_handles_labels()
    order = ["SWMM-simulated mean", "r=1 mean", "r=8 mean", "r=10 mean"]
    lookup = dict(zip(labels, handles))
    # Let tight_layout establish the actual plotting-frame geometry first;
    # the frame is not guaranteed to be centered on the raw figure canvas.
    fig.tight_layout(rect=[0.035, 0.0, 1, 0.965])
    lower_frame = axes[-1].get_position()
    frame_center_x = lower_frame.x0 + lower_frame.width / 2.0
    # Align the legend center with the plotting-frame center and remove the
    # surrounding box so it reads as a shared figure legend.
    fig.legend([lookup[label] for label in order], order, fontsize=FIGURE5_TEXT_SIZE, ncol=4,
               frameon=False, loc="upper center", bbox_to_anchor=(frame_center_x, 0.985))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / FIGURE5_NAME
    fig.savefig(figure_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    selection_path = output_dir / "Figure_5_Selected_Events_and_Errors.json"
    selection_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    contract_path = output_dir / "Figure_5_Selection_and_Aggregation_Contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "rainfall_selection": (
                    "From the eight observed 2025 held-out events, select the minimum-depth event, "
                    "the event with the maximum 25-vector ensemble-mean network-response peak, and "
                    "the maximum-depth event."
                ),
                "selection_excludes": "DSS reconstruction errors, NSE, and layout performance",
                "ineligible_events": (
                    "All development events and the synthetic 200-year design storm are excluded "
                    "from this observed held-out-event illustration."
                ),
                "curve_aggregation": (
                    "For each parameter vector, retain the 77-node trajectory; average the 25 "
                    "parameter-vector trajectories node by node before plotting the network mean."
                ),
                "band_aggregation": (
                    "Pointwise 10th-90th percentiles across the 77 nodes after parameter-ensemble averaging; "
                    "this preserves the original node-wise shadow interval."
                ),
                "error_inset": {
                    "Qp": "relative peak-flow error (%)",
                    "Tp": "absolute peak-time difference (min)",
                    "V": "relative event-volume error based on event-integrated network-mean inflow (%)",
                    "display": "three-row numeric table; units are stated in row labels",
                },
                "events": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"figure": str(figure_path), "selection": str(selection_path), "events": metrics}


def _node_risk_table(metrics: dict[str, Any], rank: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    event_maps = (
        metrics["node_event_maps"]["observed"][rank]
        + metrics["node_event_maps"]["design"][rank]
    )
    node_ids = sorted({node for event_map in event_maps for node in event_map})
    for node_id in node_ids:
        values = [float(event_map[node_id]) for event_map in event_maps
                  if node_id in event_map and np.isfinite(event_map[node_id])]
        if values:
            rows.append({"Node_ID": node_id, "Median_NSE": float(np.median(values)),
                         "Negative_NSE_Rate": float(np.mean(np.asarray(values) < 0.0)),
                         "Evaluable_Scenario_Count": len(values)})
    return pd.DataFrame(rows)


def _node_median_distributions(metrics: dict[str, Any], ranks: tuple[int, ...]) -> dict[str, dict[int, np.ndarray]]:
    """Return one median NSE per node after aggregating across events.

    Figure 6a is intended to describe the typical node-level performance.  It
    therefore first aggregates the event-wise NSE values within each node and
    only then compares the resulting node summaries across the network.  This
    avoids giving events (or nodes with more evaluable values) disproportionate
    weight, and is distinct from ``network_medians``, which takes the median
    across nodes separately within each event.
    """

    distributions: dict[str, dict[int, np.ndarray]] = {"observed": {}, "design": {}}
    for event_type in ("observed", "design"):
        for rank in ranks:
            node_medians: list[float] = []
            for values in metrics["node_lists"][event_type][rank].values():
                finite_values = np.asarray(values, dtype=float)
                finite_values = finite_values[np.isfinite(finite_values)]
                if finite_values.size:
                    node_medians.append(float(np.median(finite_values)))
            distributions[event_type][rank] = np.asarray(node_medians, dtype=float)
    return distributions


def generate_figure6(context: dict[str, Any], output_dir: Path = NODE_LEVEL_DIR, rank: int = 8) -> dict[str, Any]:
    ranks = tuple(range(1, 11))
    metrics = calculate_metrics(context, ranks)
    node_risk = _node_risk_table(metrics, rank)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), dpi=600,
                             gridspec_kw={"width_ratios": [2.10, 1.0]})
    ax = axes[0]
    # Panel a uses one value per node: median NSE across all evaluable events
    # for that node.  The old event-level network-median distribution is kept
    # below as a separate legacy diagnostic summary for reproducibility.
    distributions = _node_median_distributions(metrics, ranks)
    all_distributions = [values[np.isfinite(values)] for kind in distributions.values() for values in kind.values()]
    finite = np.concatenate([values for values in all_distributions if values.size])
    lower = min(-1.2, float(np.nanmin(finite)) * 1.06)
    set_negative_log_axis(ax, lower, 1.02)
    ax.set_ylim(lower, 1.02)
    # Keep the negative tail visible, but omit the minor -0.25 tick from the
    # manuscript-facing axis labels.  The remaining ticks still use the
    # signed-log mapping established by set_negative_log_axis.
    visible_ticks = [float(value) for value in ax.get_yticks()
                     if not np.isclose(float(value), -0.25)]
    ax.set_yticks(visible_ticks)
    ax.set_yticklabels([f"{value:.2f}".rstrip("0").rstrip(".")
                        for value in visible_ticks])
    spacing = 1.46
    bp_real = bp_design = None
    for r in ranks:
        real_values = distributions["observed"][r]
        design_values = distributions["design"][r]
        real_values = real_values[np.isfinite(real_values)]
        design_values = design_values[np.isfinite(design_values)]
        # Keep the paired observed/design boxes visually associated at each r.
        real_position = float(r) * spacing - 0.21
        design_position = float(r) * spacing + 0.21
        draw_half_violin(ax, real_values, real_position, REAL_COLOR, side="left", width=0.30,
                          y_limits=(lower, 1.02), edgecolor="none")
        draw_half_violin(ax, design_values, design_position, DESIGN_COLOR, side="right", width=0.30,
                          y_limits=(lower, 1.02), edgecolor="none")
        bp_real = ax.boxplot([real_values], positions=[real_position], widths=0.28,
                             patch_artist=True, zorder=4,
                             boxprops={"facecolor": REAL_COLOR, "edgecolor": "black", "linewidth": 1.45, "alpha": 0.90},
                             whiskerprops={"color": "black", "linewidth": 1.25},
                             capprops={"color": "black", "linewidth": 1.25},
                             medianprops={"color": "black", "linewidth": 1.55},
                             flierprops={"marker": "o", "markersize": 2.2, "markerfacecolor": REAL_COLOR,
                                         "markeredgecolor": REAL_COLOR, "alpha": 0.55})
        bp_design = ax.boxplot([design_values], positions=[design_position], widths=0.28,
                               patch_artist=True, zorder=4,
                               boxprops={"facecolor": DESIGN_COLOR, "edgecolor": "black", "linewidth": 1.45, "alpha": 0.90},
                               whiskerprops={"color": "black", "linewidth": 1.25},
                               capprops={"color": "black", "linewidth": 1.25},
                               medianprops={"color": "black", "linewidth": 1.55},
                               flierprops={"marker": "o", "markersize": 2.2, "markerfacecolor": DESIGN_COLOR,
                                           "markeredgecolor": DESIGN_COLOR, "alpha": 0.55})
    ax.set_xticks([float(r) * spacing for r in ranks])
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlim(spacing * 0.55, spacing * 10.45)
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.9, zorder=0)
    ax.set_xlabel("Retained rank, r (= number of sampled nodes)", fontsize=15)
    ax.set_ylabel("Node-level median NSE", fontsize=15)
    ax.grid(axis="y", alpha=0.15, linewidth=0.75)
    ax.legend([bp_real["boxes"][0], bp_design["boxes"][0]],
              ["Real rainfall events", "200-year designed event"],
              loc="lower right", bbox_to_anchor=(0.98, 0.010),
              borderaxespad=0.0, handletextpad=0.35, alignment="center",
              frameon=False, fontsize=11.0)
    # Anchor both panel labels to the frame's upper-left corner with the same
    # point offsets.  Offset points (rather than axes fractions) keep the
    # apparent distance identical even though the two panels have different
    # widths, and va="center" centers the glyph vertically on that anchor.
    ax.annotate("a", xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(5.0, -10.0), textcoords="offset points",
                ha="left", va="center", fontsize=15, fontweight="bold")
    _style_axis(ax)

    ax = axes[1]
    ax.scatter(node_risk["Median_NSE"], node_risk["Negative_NSE_Rate"], s=43, marker="o",
               facecolor=REAL_COLOR, edgecolor="white", linewidth=0.55, alpha=0.84,
               label="Direct-SVD Fixed", zorder=3)
    positive = node_risk.loc[node_risk["Negative_NSE_Rate"].gt(0.0)]
    for index, row in enumerate(positive.itertuples(index=False)):
        offset = 4 if index % 2 == 0 else -4
        ax.annotate(str(row.Node_ID), (float(row.Median_NSE), float(row.Negative_NSE_Rate)),
                    xytext=(offset, 3 if index % 2 == 0 else -8), textcoords="offset points",
                    ha="left" if offset > 0 else "right", va="bottom" if index % 2 == 0 else "top",
                    fontsize=9.5, color="#2F5368")
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=0.9, zorder=0)
    ax.set_xlim(-0.04, 1.02)
    ax.set_ylim(-0.012, min(0.55, max(0.10, float(node_risk["Negative_NSE_Rate"].max()) * 1.10)))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel(f"Node median NSE, r = {rank}", fontsize=15)
    ax.set_ylabel("Negative-NSE occurrence rate", fontsize=15, labelpad=3)
    ax.grid(alpha=0.15, linewidth=0.75)
    ax.annotate("b", xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(5.0, -10.0), textcoords="offset points",
                ha="left", va="center", fontsize=15, fontweight="bold")
    _style_axis(ax)
    # Match the compact manuscript layout used by Figure 8: let Matplotlib
    # solve the label/axis spacing, then trim the remaining canvas padding at
    # export without changing the plotted data or axis limits.
    fig.tight_layout(pad=0.35)
    # Move panel b's vertical title a little toward its own y-axis after the
    # layout has been solved, so the adjustment is not absorbed by tight_layout.
    axes[1].yaxis.set_label_coords(-0.11, 0.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / FIGURE6_NAME
    fig.savefig(figure_path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    risk_path = output_dir / "Figure_6_Node_Risk_r8.csv"
    node_risk.to_csv(risk_path, index=False)
    node_summary_path = output_dir / "Figure_6_Node_Median_Summary.csv"
    pd.DataFrame([
        {
            "Sensor_Count": r,
            "Median_Observed_Node_Median_NSE": float(np.nanmedian(distributions["observed"][r])),
            "Median_Design_Node_Median_NSE": float(np.nanmedian(distributions["design"][r])),
            "Median_Combined_Node_Median_NSE": float(np.nanmedian(np.concatenate([distributions["observed"][r], distributions["design"][r]]))),
            "Observed_Node_Count": int(np.isfinite(distributions["observed"][r]).sum()),
            "Design_Node_Count": int(np.isfinite(distributions["design"][r]).sum()),
        }
        for r in ranks
    ]).to_csv(node_summary_path, index=False)
    legacy_distributions = {
        "observed": {r: np.asarray(metrics["network_medians"]["observed"][r], dtype=float) for r in ranks},
        "design": {r: np.asarray(metrics["network_medians"]["design"][r], dtype=float) for r in ranks},
    }
    summary_path = output_dir / "Figure_6_Network_Median_Summary.csv"
    pd.DataFrame([
        {"Sensor_Count": r,
         "Median_Network_NSE": float(np.nanmedian(np.concatenate([legacy_distributions["observed"][r], legacy_distributions["design"][r]]))),
         "Scenario_Count": int(np.isfinite(legacy_distributions["observed"][r]).sum() + np.isfinite(legacy_distributions["design"][r]).sum())}
        for r in ranks
    ]).to_csv(summary_path, index=False)
    return {
        "figure": str(figure_path),
        "node_risk": str(risk_path),
        "node_median_summary": str(node_summary_path),
        "legacy_network_summary": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("figure4", "figure5", "figure6", "all"), required=True)
    args = parser.parse_args()
    context = load_context()
    outputs: dict[str, Any] = {}
    if args.figure in {"figure4", "all"}:
        outputs["figure4"] = generate_figure4(context)
    if args.figure in {"figure5", "all"}:
        outputs["figure5"] = generate_figure5(context)
    if args.figure in {"figure6", "all"}:
        outputs["figure6"] = generate_figure6(context)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
