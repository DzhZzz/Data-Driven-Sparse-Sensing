"""Plot the rank-aligned placement benchmark from its 9,000 plotted values."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from utils.paths import DATA_DIR, OUTPUTS_DIR


INPUT = DATA_DIR / "figure7_scenario_values.csv"
OUTPUT = OUTPUTS_DIR / "Figure_7_Rank_Aligned_Placement_Benchmark.png"
SUMMARY = OUTPUTS_DIR / "Figure_7_Summary.csv"
METHODS = ("DSS-QR", "Random placement", "Greedy D-opt", "GA-Reconstruction")
COLORS = {
    "DSS-QR": "#0072B2",
    "Random placement": "#D55E00",
    "Greedy D-opt": "#009E73",
    "GA-Reconstruction": "#CC79A7",
}
LABELS = {
    "DSS-QR": "QR",
    "Random placement": "Random placement",
    "Greedy D-opt": "Greedy D-opt",
    "GA-Reconstruction": "Genetic Algorithm",
}


def run() -> None:
    frame = pd.read_csv(INPUT)
    required = {"method", "r", "scenario_system_nse"}
    if not required.issubset(frame.columns) or len(frame) != 9000:
        raise RuntimeError("Figure 7 input must contain 4 x 10 x 225 values.")
    counts = frame.groupby(["method", "r"]).size()
    if set(frame["method"]) != set(METHODS) or not counts.eq(225).all():
        raise RuntimeError("Every Figure 7 box must contain 225 held-out simulations.")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 12.5,
            "axes.linewidth": 0.85,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(11.25, 5.728), dpi=600)
    offsets = {
        "DSS-QR": -0.285,
        "Random placement": -0.095,
        "Greedy D-opt": 0.095,
        "GA-Reconstruction": 0.285,
    }
    for method in METHODS:
        data = [
            frame.loc[
                frame["method"].eq(method) & frame["r"].eq(rank),
                "scenario_system_nse",
            ].to_numpy(dtype=float)
            for rank in range(1, 11)
        ]
        box = ax.boxplot(
            data,
            positions=np.arange(1, 11, dtype=float) + offsets[method],
            widths=0.16,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
            medianprops={"color": "#1F1F1F", "linewidth": 1.55},
            whiskerprops={"color": "black", "linewidth": 1.25},
            capprops={"color": "black", "linewidth": 1.25},
            boxprops={
                "facecolor": COLORS[method],
                "edgecolor": "black",
                "linewidth": 1.45,
            },
        )
        for patch in box["boxes"]:
            patch.set_alpha(0.72)

    grouped = frame.groupby(["method", "r"])["scenario_system_nse"]
    lower = min(-0.30, float(np.floor((grouped.quantile(0.05).min() - 0.03) * 20) / 20))
    upper = max(1.03, float(np.ceil(grouped.quantile(0.95).max() * 100) / 100))
    ax.axhline(0.0, color="0.48", linewidth=0.75, linestyle="--", zorder=0)
    ax.set_xlim(0.55, 10.45)
    ax.set_ylim(lower, upper)
    ticks = [float(value) for value in ax.get_yticks() if lower <= float(value) < 1.0]
    ax.set_yticks(sorted(set(ticks + [1.0])))
    ax.set_ylim(lower, upper)
    ax.set_xticks(np.arange(1, 11))
    ax.set_xlabel("Rank-matched sensor budget, r (retained rank = sampled nodes)", fontsize=15)
    ax.set_ylabel("System-level NSE", fontsize=15)
    ax.tick_params(axis="both", labelsize=12.5)
    ax.grid(axis="both", color="0.70", alpha=0.22, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Patch(
                facecolor=COLORS[method],
                edgecolor="black",
                linewidth=1.45,
                alpha=0.72,
                label=LABELS[method],
            )
            for method in METHODS
        ],
        loc="lower left",
        frameon=False,
        ncol=2,
        fontsize=11.0,
    )
    fig.tight_layout(pad=0.8)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = (
        frame.groupby(["method", "r"])["scenario_system_nse"]
        .agg(n="size", q05=lambda x: x.quantile(0.05), q25=lambda x: x.quantile(0.25),
             median="median", q75=lambda x: x.quantile(0.75), q95=lambda x: x.quantile(0.95))
        .reset_index()
    )
    summary.to_csv(SUMMARY, index=False)
    print(OUTPUT)


if __name__ == "__main__":
    run()
