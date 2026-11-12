"""Figure 3: observed and SWMM-simulated event-excess flow at the OF-02 flow meter (calibration E01, E04; validation E02)."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import data, plotting as P

EVENTS = ("E01", "E04", "E02")
TITLES = {"E01": "E01, 11 Jun 2024 — calibration", "E04": "E04, 15–16 Jun 2024 — calibration", "E02": "E02, 12 Jun 2024 — validation"}
TICK_HOURS = {"E01": 2, "E04": 4, "E02": 2}
BOX_ANCHOR, BOX_CLEARANCE = (0.96, 0.955), 0.06  # metrics box: upper right; curves stay 0.06 axes fraction below it


def event_metrics(event: pd.DataFrame) -> dict[str, float]:
    """NSE, peak and volume bias (%) and peak lag (min, simulated minus observed) of the event-excess flow."""
    obs, sim = event.Observed_Excess_CMS.to_numpy(float), event.Modeled_Excess_CMS.to_numpy(float)
    stamps = pd.to_datetime(event.Timestamp_Local)
    minutes = (stamps - stamps.iloc[0]).dt.total_seconds().to_numpy() / 60.0
    return {"nse": float(1.0 - np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)),
            "peak_bias_pct": float(100.0 * (sim.max() / obs.max() - 1.0)), "volume_bias_pct": float(100.0 * (sim.sum() / obs.sum() - 1.0)),
            "peak_lag_min": float(minutes[np.argmax(sim)] - minutes[np.argmax(obs)])}


def _signed(value: float, decimals: int) -> str:
    text = f"{abs(value):.{decimals}f}"
    return text if float(text) == 0.0 else ("+" if value > 0.0 else "−") + text


def _box_text(m: dict[str, float]) -> str:
    nse = f"{abs(m['nse']):.3f}"
    nse = "−" + nse if m["nse"] < 0.0 and float(nse) != 0.0 else nse
    return "\n".join([f"NSE = {nse}", f"Peak bias = {_signed(m['peak_bias_pct'], 1)}%", f"Volume bias = {_signed(m['volume_bias_pct'], 1)}%",
                      f"Peak lag = {_signed(m['peak_lag_min'], 0)} min"])


def figure3() -> dict[str, dict[str, float]]:
    P.start({"font.family": "Arial", "font.size": 7.5, "axes.titlesize": 8.0, "axes.labelsize": 8.0, "xtick.labelsize": 7.0,
             "ytick.labelsize": 7.0, "legend.fontsize": 7.5, "axes.unicode_minus": True, "pdf.fonttype": 42})
    series = data.calibration_series()
    fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.8), dpi=600)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.92, wspace=0.19)
    metrics, boxes, curves, lines = {}, {}, {}, []
    for ax, event_id in zip(axes, EVENTS):
        event = series[series.Event_ID == event_id]
        metrics[event_id] = event_metrics(event)
        times = pd.to_datetime(event.Timestamp_Local, format="%Y-%m-%d %H:%M:%S").to_numpy()
        obs, sim = event.Observed_Excess_CMS.to_numpy(float), event.Modeled_Excess_CMS.to_numpy(float)
        lines = [ax.plot(times, obs, color="#111111", linewidth=1.3, label="Observed")[0],
                 ax.plot(times, sim, color="#D62728", linewidth=1.1, label="SWMM-simulated")[0]]
        ax.set_xlim(times[0], times[-1])
        locator = mdates.HourLocator(byhour=list(range(0, 24, TICK_HOURS[event_id])))
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, zero_formats=["", "%Y", "%b", "%d %b", "%H:%M", "%H:%M"], show_offset=False))
        ax.xaxis.set_minor_locator(mdates.HourLocator())
        ax.set_title(TITLES[event_id])
        ax.set_xlabel("Local time")
        ax.grid(alpha=0.25, linewidth=0.5)
        boxes[event_id] = ax.text(*BOX_ANCHOR, _box_text(metrics[event_id]), transform=ax.transAxes, ha="right", va="top", multialignment="left",
                                  fontsize=7.0, bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none", "linewidth": 0.0},
                                  zorder=5)
        curves[event_id] = (mdates.date2num(times), (obs, sim))
    axes[0].set_ylabel("Event-excess flow at OF-02 (m³/s)")
    centre = 0.5 * (axes[0].get_position().x0 + axes[-1].get_position().x1)
    fig.legend(lines, ["Observed", "SWMM-simulated"], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(centre, 0.0))
    # where a curve would run under the metrics box, raise the top of the y axis until it clears the box
    fig.canvas.draw()
    for ax, event_id in zip(axes, EVENTS):
        display = boxes[event_id].get_bbox_patch().get_window_extent(fig.canvas.get_renderer())
        in_axes = display.transformed(ax.transAxes.inverted())
        x0 = float(ax.transData.inverted().transform((display.x0, display.y0))[0])
        x1 = float(ax.transData.inverted().transform((display.x1, display.y0))[0])
        x, ys = curves[event_id]
        inside = (x >= x0) & (x <= x1)
        highest = max(max(np.interp([x0, x1], x, y).tolist() + y[inside].tolist()) for y in ys)
        bottom, top = ax.get_ylim()
        needed = bottom + (highest - bottom) / (in_axes.y0 - BOX_CLEARANCE)
        if needed > top:
            ax.set_ylim(bottom, needed)
    fig.canvas.draw()
    P.save(fig, "Figure_3_calibration", dpi=600, facecolor="white")
    plt.close(fig)
    return metrics
