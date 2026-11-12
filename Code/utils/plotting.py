"""Plotting helpers shared by the figures (the style of the paper's figures)."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np

from utils.paths import OUTPUTS_DIR

ARIAL = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]}
BOX_RC = {**ARIAL, "font.size": 12.5, "axes.linewidth": 0.85, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "pdf.fonttype": 42}


def start(rc: dict | None = None) -> None:
    """Settings of a fresh matplotlib session plus `rc`, so that each figure is drawn independently of the others."""
    mpl.rc_file_defaults()
    mpl.rcParams.update(rc or {})


def save(fig, name: str, **kwargs) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUTS_DIR / f"{name}.png", **kwargs)


def style(ax, tick: float = 12.5) -> None:
    ax.tick_params(axis="both", labelsize=tick)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#4a4a4a")


def tag(ax, text: str, size: float = 15) -> None:
    ax.annotate(text, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(5.0, -10.0), textcoords="offset points",
                ha="left", va="center", fontsize=size, fontweight="bold")


def letter(ax, text: str) -> None:
    """Panel letter of the supplementary figures: 16 pt bold, 6 pt right of and 5 pt below the top-left corner."""
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(6, -5), textcoords="offset points", ha="left", va="top",
                fontsize=16, fontweight="bold")


# ---- y-axis rule of the placement boxes (Figure 7a): every 5th-95th percentile whisker is drawn in full
def box_ylim(q05_min: float, q95_max: float) -> tuple[float, float]:
    return min(-0.30, float(np.floor((q05_min - 0.03) * 20) / 20)), max(1.03, float(np.ceil(q95_max * 100) / 100))


def box_yticks(ax, lower: float, upper: float) -> None:
    ax.set_ylim(lower, upper)
    ticks = [float(v) for v in ax.get_yticks() if lower <= float(v) < 1.0]
    ax.set_yticks(sorted(set(ticks + [1.0])))
    ax.set_ylim(lower, upper)


# ---- signed logarithmic NSE axis (Figure 6): linear above 0, -log10(1 - NSE) below 0
def forward(values):
    array = np.asarray(values, dtype=float)
    result = np.array(array, copy=True)
    negative = array < 0.0
    result[negative] = -np.log10(1.0 - array[negative])
    return result


def inverse(values):
    array = np.asarray(values, dtype=float)
    result = np.array(array, copy=True)
    negative = array < 0.0
    result[negative] = -(10.0 ** (-array[negative]) - 1.0)
    return result


def nse_ticks(lower: float, upper: float = 1.0) -> list[float]:
    negative = [v for v in (-0.5, -1.0, -2.0, -5.0, -10.0, -20.0, -50.0) if lower <= v]
    return sorted(negative) + [v for v in (0.0, 0.25, 0.5, 0.75, 1.0) if v <= upper]


def tick_label(value, _position=None) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")
