"""Shared plotting helpers for the reconstruction figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from scipy.stats import gaussian_kde

from utils.reconstruction import projection_operators, reconstruct


REAL_COLOR = "#4a98c5"
DESIGN_COLOR = "#dc6d57"
PANEL_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14
ANNOTATION_FONT_SIZE = 13


def draw_half_violin(ax: plt.Axes, values: list[float] | np.ndarray, position: float,
                      color: str, side: str, width: float = 0.28,
                      y_limits: tuple[float, float] | None = None,
                      edgecolor: str = "black") -> None:
    values_array = np.asarray(values, dtype=float)
    values_array = values_array[np.isfinite(values_array)]
    if values_array.size < 2 or np.nanstd(values_array) == 0.0:
        return
    lower = float(np.nanmin(values_array))
    upper = float(np.nanmax(values_array))
    y_min = lower - 0.08 if y_limits is None else max(y_limits[0], lower - 0.08)
    y_max = upper + 0.08 if y_limits is None else min(y_limits[1], upper + 0.08)
    if y_max <= y_min:
        return
    transform = ax.yaxis.get_transform()
    scale_values = np.asarray(transform.transform(values_array), dtype=float)
    grid_scale = np.linspace(float(transform.transform(y_min)), float(transform.transform(y_max)), 400)
    grid = np.asarray(transform.inverted().transform(grid_scale), dtype=float)
    density = gaussian_kde(scale_values)(grid_scale)
    if not np.isfinite(density).any() or float(np.nanmax(density)) <= 0.0:
        return
    scaled = density / float(np.nanmax(density)) * width
    if side == "left":
        ax.fill_betweenx(grid, position, position - scaled, facecolor=color,
                         edgecolor=edgecolor, linewidth=1.15 if edgecolor != "none" else 0.0,
                         alpha=0.52, zorder=4)
    else:
        ax.fill_betweenx(grid, position, position + scaled, facecolor=color,
                         edgecolor=edgecolor, linewidth=1.15 if edgecolor != "none" else 0.0,
                         alpha=0.52, zorder=4)


def set_negative_log_axis(ax: plt.Axes, lower: float, upper: float = 1.1,
                           linthresh: float = 1.0) -> None:
    """Use a linear positive NSE axis and a logarithmically compressed negative tail.

    The positive branch is deliberately untouched (``f(y)=y`` for ``y>=0``),
    while the negative branch uses ``-log10(1-y)``.  This preserves resolution
    for the 0--1 performance distributions and prevents a few extreme negative
    NSE values from consuming the whole panel.
    """
    del linthresh  # retained for compatibility with older callers

    def _forward(values: object) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        result = np.array(array, copy=True)
        negative = array < 0.0
        result[negative] = -np.log10(1.0 - array[negative])
        return result

    def _inverse(values: object) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        result = np.array(array, copy=True)
        negative = array < 0.0
        result[negative] = -(10.0 ** (-array[negative]) - 1.0)
        return result

    ax.set_yscale("function", functions=(_forward, _inverse))
    negative_candidates = (-0.25, -0.5, -1.0, -2.0, -5.0, -10.0, -100.0, -1000.0)
    negative = []
    for value in negative_candidates:
        if lower <= value < 0.0 and value not in negative:
            negative.append(value)
    positive = [value for value in (0.0, 0.25, 0.50, 0.75, 1.0) if value <= upper]
    values = negative + positive
    ax.set_yticks(values)
    ax.set_yticklabels([f"{value:.2f}".rstrip("0").rstrip(".") for value in values])


def plot_scatter_panel(ax: plt.Axes, true_values: np.ndarray, predicted: np.ndarray,
                        label: str, panel_tag: str, show_ylabel: bool,
                        axis_bounds: tuple[float, float] | None = None) -> Any:
    true_flat = np.asarray(true_values, dtype=float).reshape(-1)
    pred_flat = np.asarray(predicted, dtype=float).reshape(-1)
    if axis_bounds is None:
        lower = float(min(np.nanmin(true_flat), np.nanmin(pred_flat)))
        upper = float(max(np.nanmax(true_flat), np.nanmax(pred_flat)))
        padding = (upper - lower) * 0.03 if upper > lower else 1.0
        lower -= padding
        upper += padding
    else:
        lower, upper = (float(axis_bounds[0]), float(axis_bounds[1]))
    hexbin = ax.hexbin(true_flat, pred_flat, gridsize=52, mincnt=1, cmap="RdBu_r",
                       norm=LogNorm(), linewidths=0, extent=(lower, upper, lower, upper))
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="#2f2f2f", linewidth=1.0)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Simulated Flow (m³/s)", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=8)
    ax.set_ylabel("Reconstructed Flow (m³/s)" if show_ylabel else "",
                  fontsize=AXIS_LABEL_FONT_SIZE, labelpad=8)
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(True, alpha=0.14, linewidth=0.7)
    ax.text(0.045, 0.955, panel_tag, transform=ax.transAxes, ha="left", va="top",
            fontsize=PANEL_FONT_SIZE, fontweight="bold")
    ax.text(0.992, 0.012, label, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=ANNOTATION_FONT_SIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#4a4a4a")
    return hexbin


def plot_system_combined(metrics: dict[str, Any], context: dict[str, Any], ranks: tuple[int, ...],
                         output_path: Path, top_kind: str,
                         scatter_ranks: tuple[int, ...] | None = None) -> Path:
    representative = list(dict.fromkeys(
        tuple(int(rank) for rank in (scatter_ranks if scatter_ranks is not None else (ranks[0], ranks[len(ranks) // 2], ranks[-1])))
    ))
    if any(rank not in ranks for rank in representative) or len(representative) != 3:
        raise ValueError(f"scatter_ranks must contain exactly three ranks from {ranks}: {representative}")
    operators = projection_operators(context["basis"], context["sensor_sets"])
    true_parts: dict[int, list[np.ndarray]] = {rank: [] for rank in representative}
    pred_parts: dict[int, list[np.ndarray]] = {rank: [] for rank in representative}
    for event in context["events"]:
        observed = np.asarray(event["data"], dtype=float)
        for rank in representative:
            predicted = reconstruct(observed, operators[rank], context["sensor_sets"][rank], context["scale"])
            true_parts[rank].append(observed)
            pred_parts[rank].append(predicted)

    fig = plt.figure(figsize=(15, 11.2), dpi=300)
    outer = fig.add_gridspec(2, 4, height_ratios=[1.65, 1.35], width_ratios=[1, 1, 1, 0.04],
                             left=0.09, right=0.985, top=0.975, bottom=0.08, hspace=0.13, wspace=0.08)
    ax_top = fig.add_subplot(outer[0, 0:3])
    bottom_axes = [fig.add_subplot(outer[1, index]) for index in range(3)]
    colorbar_ax = fig.add_subplot(outer[1, 3])
    if top_kind == "boxplot":
        spacing = 2.0
        legend_real = legend_design = None
        for rank in ranks:
            bp1 = ax_top.boxplot([metrics["system"]["observed"][rank]], positions=[rank * spacing - 0.4],
                                 patch_artist=True, widths=0.6, medianprops={"color": "black"})
            bp2 = ax_top.boxplot([metrics["system"]["design"][rank]], positions=[rank * spacing + 0.4],
                                 patch_artist=True, widths=0.6, medianprops={"color": "black"})
            for patch in bp1["boxes"]:
                patch.set_facecolor(REAL_COLOR)
            for patch in bp2["boxes"]:
                patch.set_facecolor(DESIGN_COLOR)
            if legend_real is None:
                legend_real = bp1["boxes"][0]
                legend_design = bp2["boxes"][0]
        ax_top.set_xticks([rank * spacing for rank in ranks])
        ax_top.set_xticklabels(ranks)
        ax_top.set_ylim(0.5, 1.0)
        ax_top.yaxis.set_major_locator(MultipleLocator(0.05))
        top_legend = [legend_real, legend_design]
        top_legend_labels = ["Real rainfall events", "200-year designed event"]
    else:
        rank_array = np.asarray(ranks, dtype=float)
        real = metrics["system"]["observed"]
        design = metrics["system"]["design"]
        real_lowers = np.asarray([np.min(real[r]) for r in ranks], dtype=float)
        real_uppers = np.asarray([np.max(real[r]) for r in ranks], dtype=float)
        real_means = np.asarray([np.mean(real[r]) for r in ranks], dtype=float)
        design_means = np.asarray([np.mean(design[r]) for r in ranks], dtype=float)
        ax_top.fill_between(rank_array, [np.min(real[r]) for r in ranks], [np.max(real[r]) for r in ranks],
                            color=REAL_COLOR, alpha=0.18, linewidth=0, label="Real rainfall events range")
        ax_top.plot(rank_array, real_means, color=REAL_COLOR, linewidth=2.8, marker="o",
                    markersize=6, label="Real rainfall events mean")
        ax_top.plot(rank_array, design_means, color=DESIGN_COLOR, linewidth=2.8, marker="s",
                    markersize=5.6, label="200-year designed event")
        x_span = float(np.ptp(rank_array)) if rank_array.size > 1 else 1.0
        x_padding = max(0.12, 0.02 * x_span)
        ax_top.set_xlim(rank_array.min() - x_padding, rank_array.max() + x_padding)
        y_top = float(np.nanmax(np.concatenate([real_uppers, design_means])))
        y_padding = max(0.02, 0.03 * max(y_top, 1.0))
        # Keep a visible lower margin below the r=1 uncertainty band.  The
        # previous -0.2 bound left the shaded range nearly coincident with the
        # bottom spine, making the negative NSE extent visually cramped.
        ax_top.set_ylim(-0.30, y_top + y_padding)
        ax_top.set_xticks(rank_array)
        ax_top.yaxis.set_major_locator(MultipleLocator(0.2))
        top_legend = None
        top_legend_labels = None
    ax_top.set_xlabel("Retained rank, r (= number of sampled nodes)", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=8)
    ax_top.set_ylabel("System-level NSE", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=8)
    ax_top.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax_top.grid(axis="y", alpha=0.16, linewidth=0.8)
    for spine in ax_top.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#4a4a4a")
    if top_legend is None:
        ax_top.legend(fontsize=LEGEND_FONT_SIZE, loc="lower right", frameon=False)
    else:
        ax_top.legend(top_legend, top_legend_labels, fontsize=LEGEND_FONT_SIZE, loc="lower right",
                      frameon=True, edgecolor="#666666")
    ax_top.text(0.015, 0.955, "a", transform=ax_top.transAxes, fontsize=PANEL_FONT_SIZE,
                fontweight="bold", va="top")

    all_true = np.concatenate([np.asarray(part, dtype=float).reshape(-1) for rank in representative for part in true_parts[rank]])
    all_pred = np.concatenate([np.asarray(part, dtype=float).reshape(-1) for rank in representative for part in pred_parts[rank]])
    common_lower = float(min(np.nanmin(all_true), np.nanmin(all_pred)))
    common_upper = float(max(np.nanmax(all_true), np.nanmax(all_pred)))
    common_padding = (common_upper - common_lower) * 0.03 if common_upper > common_lower else 1.0
    common_bounds = (common_lower - common_padding, common_upper + common_padding)
    all_event_values = {
        rank: metrics["system"]["observed"][rank] + metrics["system"]["design"][rank]
        for rank in representative
    }
    last_hex = None
    for index, rank in enumerate(representative):
        true_values = np.concatenate(true_parts[rank], axis=1)
        pred_values = np.concatenate(pred_parts[rank], axis=1)
        event_values = all_event_values[rank]
        label = f"r={rank}, mean simulation NSE={np.mean(event_values):.3f}"
        last_hex = plot_scatter_panel(
            bottom_axes[index], true_values, pred_values, label, f"b{index + 1}", index == 0,
            axis_bounds=common_bounds,
        )
    if last_hex is not None:
        colorbar = fig.colorbar(last_hex, cax=colorbar_ax)
        colorbar.set_label("Counts", fontsize=LEGEND_FONT_SIZE)
        colorbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    # Keep the vertical y-axis titles on the top and lower rows on the same
    # horizontal reference line, despite their different subplot widths.
    fig.align_ylabels([ax_top, bottom_axes[0]])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    return output_path
