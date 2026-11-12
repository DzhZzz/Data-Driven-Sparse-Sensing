"""Figure 8 (measurement errors and sensor loss), Figure S3 (all measurement-error cases) and Figure S4 (sensor loss for every m)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from utils import data, plotting as P
from utils.paths import OUTPUTS_DIR

NOISE = [("clean", "Noise-free", "#bdbdbd"), ("5pct", "±5% multiplicative noise", "#c6dbef"), ("10pct", "±10% multiplicative noise", "#7fb0d8"),
         ("15pct", "±15% multiplicative noise", "#3b78b5")]
BOX_UNC, BOX_REFIT, BOX_TRUNC, EDGE, RED, REF_GREY = "#8fb1cc", "#a9d5a0", "#f2a69c", "#222222", "#d65244", "#555555"
LOSS_BOXES = [("uncompensated", "Uncompensated", BOX_UNC), ("refit", "Refitted ridge", BOX_REFIT), ("truncated", "Truncated basis", BOX_TRUNC)]
LOSS_LINES = {"uncompensated": ("Uncompensated", "#2a7ab0", "o"), "refit": ("Refitted ridge", "#58a55c", "s"),
              "truncated": ("Truncated basis", RED, "^")}
ERROR_LINES = [("clean", "Noise-free", "#222222", "-", "o", True),  # Figure 8a: one representative level of each error type
               ("15pct", "±15% multiplicative noise", "#3b78b5", "-", "v", True),
               ("additive_5Ls", "Additive noise, σ = 5 L/s", "#2a9d8f", "--", "o", False),
               ("additive_10Ls", "Additive noise, σ = 10 L/s", "#2a9d8f", "-", "o", True),
               ("detection_floor_10Ls", "Detection floor, 10 L/s", "#7b5ea7", ":", "D", True),
               ("bias_all_minus20pct", "All sensors, −20% bias", "#e9a23b", "-", "^", True),
               ("bias_all_plus20pct", "All sensors, +20% bias", "#b2182b", "-", "s", True)]


def error_values(system: pd.DataFrame) -> dict[str, np.ndarray]:
    """Observed-storm values of every measurement-error case, aligned with the error-free QR-DSS values ('clean_mXX')."""
    errors = data.measurement_errors()
    observed = errors["rainfall_types"] == "observed"
    order = pd.Index(errors["scenario_ids"])
    values = {}
    for m in data.RANKS:
        clean = system[system.m == m].set_index("scenario_id").system_nse.loc[order].to_numpy()
        values[f"clean_m{m:02d}"] = clean[observed]
    for key, array in errors.items():
        if array.ndim >= 1 and array.shape[-1] == len(order) and array.dtype.kind == "f":
            values[key] = array[..., observed]
    return values


def loss_values(loss: pd.DataFrame) -> dict[tuple[str, int, str], np.ndarray]:
    """Observed-storm loss per (variant, m, failed conduit)."""
    observed = loss[loss.rainfall_type == "observed"]
    return {(v, int(m), link): g.loss.to_numpy() for (v, m, link), g in observed.groupby(["variant", "m", "failed_link"])}


def loss_summary(ctx: dict, losses: dict) -> pd.DataFrame:
    """Mean and largest (worst sensor) per-sensor median loss for m = 2-10, observed storms."""
    rows = []
    for variant in LOSS_LINES:
        for m in range(2, 11):
            medians = [float(np.median(losses[(variant, m, ctx["links"][s])])) for s in ctx["layouts"]["QR_DSS"][m]]
            rows.append({"variant": variant, "m": m, "mean_median_loss": float(np.mean(medians)), "worst_median_loss": float(np.max(medians)),
                         "worst_sensor": ctx["links"][ctx["layouts"]["QR_DSS"][m][int(np.argmax(medians))]]})
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUTS_DIR / "Summary_sensor_loss.csv", index=False)
    return table


def _style(ax, tick=12):
    P.style(ax, tick)


def _tag(ax, text):
    ax.text(0.012, 0.965, text, transform=ax.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")


def loss_top(losses: dict) -> float:
    """Upper y limit of the loss boxes, shared by Figure 8b and Figure S4: every loss at m = 8 plus a margin, rounded up to 0.05."""
    return float(np.ceil((max(float(v.max()) for (_, m, _), v in losses.items() if m == 8) + 0.06) / 0.05) * 0.05)


def _loss_boxes(ax, layout, losses, m, top, legend=None):
    """Loss of each monitored conduit with the uncompensated, refitted-ridge and truncated-basis decoders on a linear axis up to `top`; a box
    whose median lies above the axis is marked at the top with its median."""
    for k, link in enumerate(layout, start=1):
        for (variant, _, fill), offset in zip(LOSS_BOXES, (-0.26, 0.0, 0.26)):
            values = losses[(variant, m, link)]
            ax.boxplot([values], positions=[k + offset], widths=0.22, patch_artist=True, zorder=3,
                       boxprops={"facecolor": fill, "edgecolor": EDGE, "linewidth": 1.2}, whiskerprops={"color": EDGE, "linewidth": 1.1},
                       capprops={"color": EDGE, "linewidth": 1.1}, medianprops={"color": EDGE, "linewidth": 2.4},
                       flierprops={"marker": "o", "markersize": 2.6, "markerfacecolor": "#6b6b6b", "markeredgecolor": "none", "alpha": 0.5})
            median = float(np.median(values))
            if median > top:
                y = top - 0.06 * (top + 0.05)
                ax.plot([k + offset], [y], marker="^", markersize=8, color=fill, markeredgecolor=EDGE, markeredgewidth=0.8, zorder=5)
                ax.annotate(f"{median:.0f}" if median >= 10 else f"{median:.1f}", (k + offset, y), xytext=(6, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=10, color="#333333", zorder=6,
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.9})
    ax.axhline(0.0, color=REF_GREY, linestyle=(0, (5, 3)), linewidth=1.3, zorder=2)
    ax.set_ylim(-0.05, top)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xticks(range(1, len(layout) + 1))
    ax.set_xticklabels(layout)
    ax.set_xlim(0.45, len(layout) + 0.55)
    ax.grid(axis="y", color="#f0f0f0", linewidth=1.0, zorder=0)
    if legend is not None:
        ax.legend([Patch(facecolor=f, edgecolor=EDGE) for _, _, f in LOSS_BOXES] + [Line2D([], [], color=REF_GREY, linestyle=(0, (5, 3)), linewidth=1.3)],
                  [label for _, label, _ in LOSS_BOXES] + ["No-loss reference"], frameon=False, fontsize=11, ncol=2, **legend)
    _style(ax)


def _medians(values: dict, key: str) -> np.ndarray:
    return np.array([np.median(values[f"{key}_m{m:02d}"]) for m in data.RANKS])


def _error_lines(ax, values: dict) -> None:
    """Median system NSE (observed storms) under one representative level of each type of measurement error."""
    for key, label, color, ls, marker, filled in ERROR_LINES:
        ax.plot(list(data.RANKS), _medians(values, key), color=color, linestyle=ls, linewidth=2.0, marker=marker, markersize=6.0,
                markerfacecolor=color if filled else "white", markeredgecolor=color, markeredgewidth=1.3, label=label, zorder=3)
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.6, 10.4)
    ax.set_ylim(0.72, 1.005)
    ax.grid(color="#f0f0f0", linewidth=1.0, zorder=0)
    ax.legend(loc="lower right", ncol=2, frameon=False, fontsize=11.5, columnspacing=1.6, handlelength=2.6, handletextpad=0.5)


def _noise_boxes(ax, values: dict) -> None:
    """System NSE (observed storms) for clean measurements and multiplicative noise with +/-5, 10 and 15% three-sigma envelopes."""
    lows = []
    for (case, _, color), offset in zip(NOISE, np.linspace(-0.3, 0.3, 4)):
        for m in data.RANKS:
            v = values[f"{case}_m{m:02d}"]
            lows.append(v.min())
            ax.boxplot([v], positions=[m + offset], widths=0.17, patch_artist=True, zorder=3,
                       boxprops={"facecolor": color, "edgecolor": "black", "linewidth": 1.2}, whiskerprops={"color": "black", "linewidth": 1.1},
                       capprops={"color": "black", "linewidth": 1.1}, medianprops={"color": "black", "linewidth": 1.4},
                       flierprops={"marker": "o", "markersize": 2.6, "markerfacecolor": color, "markeredgecolor": "none"})
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([str(m) for m in range(1, 11)])
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(np.floor((min(lows) - 0.01) / 0.05) * 0.05, 1.005)
    ax.grid(axis="y", color="#f0f0f0", linewidth=1.0, zorder=0)
    ax.legend([Patch(facecolor=c, edgecolor="black") for _, _, c in NOISE], [label for _, label, _ in NOISE], loc="lower right", frameon=False,
              fontsize=11.5)


def _loss_lines(ax, summary: pd.DataFrame) -> None:
    handles = []
    for variant, (label, color, marker) in LOSS_LINES.items():
        sub = summary[summary.variant == variant].sort_values("m")
        for column, ls, fill, suffix in (("mean_median_loss", "-", color, "mean"), ("worst_median_loss", "--", "white", "worst sensor")):
            line, = ax.plot(sub.m, np.maximum(sub[column].to_numpy(dtype=float), 1e-4), color=color, linestyle=ls, linewidth=2.0, marker=marker,
                            markersize=7, markerfacecolor=fill, markeredgecolor=color, markeredgewidth=1.3, zorder=3, label=f"{label}, {suffix}")
            handles.append(line)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 500)
    ax.axhline(1.0, color="#9a9a9a", linestyle=":", linewidth=1.0, zorder=1)
    ax.set_xticks(range(2, 11))
    ax.set_xlim(1.6, 10.4)
    ax.grid(color="#f0f0f0", linewidth=1.0, zorder=0)
    ax.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=10.5, columnspacing=1.2, handlelength=2.6)


def figure8(ctx: dict, values: dict, losses: dict, summary: pd.DataFrame) -> None:
    """(a) Median system NSE under representative measurement errors; (b) loss of one monitored conduit at m = 8 with the uncompensated,
    refitted-ridge and truncated-basis decoders (the paper adds a location map of these conduits, drawn from the SWMM model); (c) mean and
    worst-sensor median loss against m."""
    P.start(P.ARIAL)
    a_h, b_h, c_h, gap, top_in, bottom_in = 3.72, 3.53, 3.35, 0.61, 0.20, 0.61
    height = top_in + a_h + b_h + c_h + 2 * gap + bottom_in
    fig = plt.figure(figsize=(12.0, height), dpi=600)
    grid = fig.add_gridspec(3, 1, height_ratios=[a_h, b_h, c_h], hspace=gap / ((a_h + b_h + c_h) / 3),
                            left=0.075, right=0.985, top=1 - top_in / height, bottom=bottom_in / height)
    ax = fig.add_subplot(grid[0])
    _error_lines(ax, values)
    ax.set_xlabel("Number of monitored conduits, m", fontsize=14)
    ax.set_ylabel("Median system-level NSE", fontsize=14)
    _tag(ax, "a")
    _style(ax)

    layout = [ctx["links"][s] for s in ctx["layouts"]["QR_DSS"][8]]
    ax = fig.add_subplot(grid[1])
    _loss_boxes(ax, layout, losses, 8, loss_top(losses), {"loc": "upper left", "bbox_to_anchor": (0.12, 1.0)})
    ax.set_xlabel("Dropped conduit (QR selection order)", fontsize=14)
    ax.set_ylabel("System-level NSE loss", fontsize=14)
    _tag(ax, "b  m = 8")

    ax = fig.add_subplot(grid[2])
    _loss_lines(ax, summary)
    ax.set_xlabel("Number of monitored conduits, m", fontsize=14)
    ax.set_ylabel("System-level NSE loss", fontsize=14)
    _tag(ax, "c")
    _style(ax)
    P.save(fig, "Figure_8_robustness", dpi=600, facecolor="white")
    plt.close(fig)


def figure_s3(values: dict) -> None:
    """(a) System NSE for clean measurements and multiplicative noise; (b) additive Gaussian noise of 0.5-10 L/s (band: Q10-Q90 at
    10 L/s); (c) detection floor, +/-20% bias on all sensors or on the worst single sensor, and +/-15% multiplicative noise. Observed
    storms; medians in (b) and (c)."""
    P.start(P.ARIAL)
    fig = plt.figure(figsize=(13.0, 9.6), dpi=600)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.28, wspace=0.17, left=0.068, right=0.988, top=0.985, bottom=0.07)
    ms = np.arange(1, 11)
    ax = fig.add_subplot(grid[0, :])
    _noise_boxes(ax, values)
    ax.set_ylabel("System-level NSE", fontsize=15)
    P.letter(ax, "a")
    axes = [ax, fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    clean = _medians(values, "clean")
    ax = axes[1]
    q = lambda p: np.array([np.quantile(values[f"additive_10Ls_m{m:02d}"], p) for m in ms])  # noqa: E731
    ax.fill_between(ms, q(0.10), q(0.90), color="#2a9d8f", alpha=0.15, linewidth=0, label="10 L/s: Q10–Q90", zorder=1)
    ax.plot(ms, clean, color="#222222", linewidth=2.4, marker="o", markersize=5.5, label="0 (noise-free)", zorder=4)
    for key, label, color, ls, filled in (("additive_0p5Ls", "0.5", "#a8dcd5", "-", True), ("additive_1Ls", "1", "#7cc8bf", "-", True),
                                          ("additive_2Ls", "2", "#4fb3a7", "-", True), ("additive_5Ls", "5", "#2a9d8f", "--", False),
                                          ("additive_10Ls", "10", "#2a9d8f", "-", True)):  # 5 and 10 L/s drawn as in Figure 8a
        ax.plot(ms, _medians(values, key), color=color, linestyle=ls, linewidth=2.0, marker="o", markersize=5,
                markerfacecolor=color if filled else "white", markeredgecolor=color, markeredgewidth=1.2, label=label, zorder=3)
    ax.set_ylabel("System-level NSE (median)", fontsize=15)
    ax.legend(title="Additive noise per sensor, σ (L/s)", title_fontsize=10.5, loc="lower left", ncol=2, frameon=False, fontsize=10.5,
              handletextpad=0.4)
    P.letter(ax, "b")
    ax = axes[2]
    worst = lambda key: np.array([np.median(values[f"{key}_m{m:02d}"], axis=-1).min() for m in ms])  # noqa: E731
    for label, y, color, ls, marker, filled in (
            ("Noise-free", clean, "#222222", "-", "o", True), ("±15% multiplicative noise", _medians(values, "15pct"), "#3b78b5", "-", "v", True),
            ("Detection floor, 10 L/s", _medians(values, "detection_floor_10Ls"), "#7b5ea7", ":", "D", True),
            ("All sensors, −20% bias", _medians(values, "bias_all_minus20pct"), "#e9a23b", "-", "^", True),
            ("All sensors, +20% bias", _medians(values, "bias_all_plus20pct"), "#b2182b", "-", "s", True),
            ("Worst single sensor, −20%", worst("bias_one_minus20pct"), "#e9a23b", "--", "^", False),
            ("Worst single sensor, +20%", worst("bias_one_plus20pct"), "#b2182b", "--", "s", False)):
        ax.plot(ms, y, color=color, linestyle=ls, linewidth=2.0, marker=marker, markersize=5.5, markerfacecolor=color if filled else "white",
                markeredgecolor=color, markeredgewidth=1.2, label=label, zorder=3)
    ax.legend(loc="lower right", frameon=False, fontsize=10.5, handletextpad=0.4)
    P.letter(ax, "c")
    for ax in axes:
        ax.set_xlabel("Number of monitored conduits, m", fontsize=15)
        _style(ax, 12.5)
    for ax in axes[1:]:
        ax.set_xticks(ms)
        ax.set_xlim(0.6, 10.4)
        ax.grid(color="#f0f0f0", linewidth=1.0, zorder=0)
    P.save(fig, "Figure_S3_measurement_errors", dpi=600, facecolor="white")
    plt.close(fig)


def figure_s4(ctx: dict, losses: dict) -> None:
    """Loss of each monitored conduit for every m except 8 (Figure 8b) with the uncompensated, refitted-ridge and truncated-basis decoders, on
    the y-axis of Figure 8b."""
    P.start(P.ARIAL)
    ms = [2, 3, 4, 5, 6, 7, 9, 10]
    top = loss_top(losses)
    fig, axes = plt.subplots(4, 2, figsize=(12.0, 14.0), dpi=600)
    for k, (ax, m) in enumerate(zip(axes.ravel(), ms)):
        layout = [ctx["links"][s] for s in ctx["layouts"]["QR_DSS"][m]]
        _loss_boxes(ax, layout, losses, m, top, {"loc": "upper right"} if k == 0 else None)
        ax.set_xticklabels([s.replace("Link-", "") for s in layout], fontsize=10.5)
        _tag(ax, f"m = {m}")
        if k % 2 == 0:
            ax.set_ylabel("System-level NSE loss", fontsize=13)
        if k >= 6:
            ax.set_xlabel("Dropped conduit (QR selection order)", fontsize=13)
    fig.tight_layout(h_pad=1.2, w_pad=1.5)
    P.save(fig, "Figure_S4_sensor_loss", dpi=600, facecolor="white")
    plt.close(fig)
