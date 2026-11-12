"""Figure S5: singular-value spectrum and cumulative energy of the event-weighted development matrix."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from utils import plotting as P

R_MARK = 5  # the five modes that hold 99.2% of the energy


def figure_s5(ctx: dict) -> np.ndarray:
    """(a) Singular values normalized by the leading one, s_k / s_1, on a log axis; (b) cumulative energy share. Returns (b) in %."""
    P.start(P.BOX_RC)
    energy = ctx["energies"]  # s_k^2 / N
    k = np.arange(1, energy.size + 1)
    cumulative = np.cumsum(energy) / energy.sum() * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=600)
    ax = axes[0]
    ax.plot(k, np.sqrt(energy / energy[0]), marker="o", markersize=5, linewidth=2.0, color="#1f4e79", label="Normalized singular value", zorder=3)
    ax.axvline(R_MARK, color="r", linestyle="--", linewidth=1.6, label=f"r = {R_MARK}", zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 4.0)
    ax.set_xlabel("Mode number, k", fontsize=15)
    ax.set_ylabel("Normalized singular value", fontsize=15)
    ax.legend(loc="upper right", frameon=False, fontsize=10.5, handletextpad=0.4)
    ax = axes[1]
    at_r = cumulative[R_MARK - 1]
    ax.plot(k, cumulative, marker="o", markersize=5, linewidth=2.0, color="#1f4e79", label="Cumulative energy share", zorder=3)
    ax.axvline(R_MARK, color="r", linestyle="--", linewidth=1.6, label=f"r = {R_MARK}", zorder=2)
    ax.axhline(at_r, color="g", linestyle="--", linewidth=1.6, label=f"{at_r:.1f}% at r = {R_MARK}", zorder=2)
    ax.annotate(f"({R_MARK}, {at_r:.1f}%)", xy=(R_MARK, at_r), xytext=(24, -36), textcoords="offset points", fontsize=10.5,
                arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.0})
    ax.set_ylim(max(0.0, float(cumulative[0]) - 5.0), 102.0)
    ax.set_xlabel("Number of retained modes, r", fontsize=15)
    ax.set_ylabel("Cumulative energy share (%)", fontsize=15)
    ax.legend(loc="lower right", frameon=False, fontsize=10.5, handletextpad=0.4)
    for ax, text in zip(axes, "ab"):
        ax.set_xlim(0, 77)
        ax.tick_params(axis="both", labelsize=12.5)
        ax.grid(axis="both", color="0.70", alpha=0.22, linewidth=0.55)
        ax.set_axisbelow(True)
        P.letter(ax, text)
    fig.tight_layout(pad=0.8, w_pad=1.2)
    P.save(fig, "Figure_S5_spectrum", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return cumulative
