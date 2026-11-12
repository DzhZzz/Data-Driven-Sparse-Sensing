"""Figure 7 (placement benchmarks and decoders, observed storms) and Figure S2 (200-year design storm, condition numbers, ridge
weighting)."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from utils import data, plotting as P
from utils.core import condition_number
from utils.evaluation import conduit_metrics, models, ridge, snapshot_equal_dss, system_nse_table
from utils.paths import OUTPUTS_DIR

RANKS = list(data.RANKS)
QR_BLUE, RANDOM_RED, GA_PINK, AREA_GRAY, WEIGHT_ORANGE = "#0072B2", "#D55E00", "#CC79A7", "#7f7f7f", "#E69F00"
SITE_LINES = [("QR_DSS", QR_BLUE, "-", "o", True), ("QR_RidgeW", QR_BLUE, "--", "o", False), ("Area_RidgeW", AREA_GRAY, "--", "s", False)]
BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES = 20261018, 20000


def scenario_table(ctx: dict) -> pd.DataFrame:
    """Per-scenario system NSE of every method: DSS decoder at the QR, GA-best and Area sites, ridge decoders at the QR and Area
    sites, QR-DSS with the snapshot-equal basis, and Uniform Random (median over the candidate layouts; from Data/)."""
    by_method = models(ctx)
    by_method["QR_DSS_snapshot_equal"] = snapshot_equal_dss(ctx)
    table = system_nse_table(ctx, by_method)
    random = data.random_placement().rename(columns={"rainfall_event": "catalog_id"}).assign(method="Random_median")
    table = pd.concat([table, random[table.columns]], ignore_index=True)
    table.groupby(["method", "rainfall_type", "m"]).system_nse.agg(
        median="median", q10=lambda v: np.quantile(v, 0.1), q90=lambda v: np.quantile(v, 0.9), n="size").reset_index().to_csv(
        OUTPUTS_DIR / "Summary_system_nse_by_method.csv", index=False)
    return table


def paired_effects(table: pd.DataFrame) -> pd.DataFrame:
    """Observed storms: mean over the 8 storms of the storm median of the scenario-wise paired difference, with a 95% interval from
    20,000 bootstrap replicates that resample the storms (the same replicates for every contrast)."""
    observed = table[table.rainfall_type == "observed"]
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 8, size=(BOOTSTRAP_REPLICATES, 8))
    rows = []
    for first, second in (("QR_DSS", "GA_best_DSS"), ("QR_RidgeW", "Area_RidgeW"), ("QR_DSS", "QR_RidgeW"), ("QR_RidgeW", "QR_Ridge"),
                          ("QR_DSS", "QR_DSS_snapshot_equal")):
        for m in RANKS:
            a = observed[(observed.method == first) & (observed.m == m)].set_index("scenario_id")
            b = observed[(observed.method == second) & (observed.m == m)].set_index("scenario_id").loc[a.index]
            delta = (a.system_nse - b.system_nse).groupby(a.catalog_id).median().sort_index().to_numpy()
            boot = delta[draws].mean(axis=1)
            rows.append({"first": first, "second": second, "m": m, "mean_event_median_delta": float(delta.mean()),
                         "ci_lower": float(np.quantile(boot, 0.025)), "ci_upper": float(np.quantile(boot, 0.975))})
    effects = pd.DataFrame(rows)
    effects.to_csv(OUTPUTS_DIR / "Summary_paired_differences.csv", index=False)
    return effects


def _m_axis(ax, xlim: tuple[float, float]) -> None:
    ax.set_xticks(RANKS)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Number of monitored conduits, m", fontsize=15)
    ax.grid(alpha=0.15, linewidth=0.75)
    P.style(ax)


def _placement_boxes(ax, table: pd.DataFrame) -> None:
    """System NSE of QR, Uniform Random and the genetic algorithm (observed storms); whiskers at the 5th and 95th percentiles."""
    observed = table[table.rainfall_type == "observed"]
    methods = [("QR_DSS", "QR", QR_BLUE), ("Random_median", "Uniform Random", RANDOM_RED), ("GA_best_DSS", "Genetic Algorithm", GA_PINK)]
    values = {(method, m): observed[(observed.method == method) & (observed.m == m)].system_nse.to_numpy() for method, _, _ in methods for m in RANKS}
    lower, upper = P.box_ylim(min(float(np.percentile(v, 5)) for v in values.values()), max(float(np.percentile(v, 95)) for v in values.values()))
    for (method, _, color), offset in zip(methods, (-0.26, 0.0, 0.26)):
        for m in RANKS:
            ax.boxplot([values[(method, m)]], positions=[m + offset], widths=0.22, whis=(5, 95), showfliers=False, patch_artist=True,
                       manage_ticks=False, zorder=3, boxprops={"facecolor": to_rgba(color, 0.78), "edgecolor": "#222222", "linewidth": 1.7},
                       whiskerprops={"color": "black", "linewidth": 1.5}, capprops={"color": "black", "linewidth": 1.5},
                       medianprops={"color": "black", "linewidth": 2.4})
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xticks(RANKS)
    ax.set_xticklabels([str(m) for m in RANKS])
    P.box_yticks(ax, lower, upper)
    ax.set_ylabel("System-level NSE", fontsize=15)
    ax.legend([Patch(facecolor=to_rgba(c, 0.78), edgecolor="#222222", linewidth=1.2) for _, _, c in methods], [label for _, label, _ in methods],
              loc="lower left", ncol=3, frameon=False, fontsize=12)


def _site_medians(ax, table: pd.DataFrame, domain: str, legend: dict) -> None:
    """Median system NSE at the QR and Area sites with the DSS and weighted ridge decoders; Area-DSS medians below 0.5 are drawn as
    triangles at the bottom of the axis, with their values."""
    medians = table.groupby(["method", "rainfall_type", "m"]).system_nse.median()
    floor, triangle_y = 0.5, 0.47
    for method, color, ls, marker, filled in SITE_LINES:
        ax.plot(RANKS, medians.loc[(method, domain)].loc[RANKS].to_numpy(), color=color, linestyle=ls, linewidth=2.0, marker=marker,
                markersize=6.5, markerfacecolor=color if filled else "white", markeredgecolor=color, markeredgewidth=1.3, zorder=3)
    y = medians.loc[("Area_DSS", domain)].loc[RANKS].to_numpy()
    x = np.asarray(RANKS)
    inside = y >= floor
    ax.plot(x[inside], y[inside], linestyle="none", marker="s", markersize=7, color=AREA_GRAY, zorder=3)
    ax.plot(x[~inside], np.full((~inside).sum(), triangle_y), linestyle="none", marker="v", markersize=8, color=AREA_GRAY, zorder=3)
    for m, value in zip(x[~inside], y[~inside]):
        text = (f"{value:.2f}" if abs(value) < 1 else f"{value:.1f}").replace("-", "−")
        ax.annotate(text, (m, triangle_y), xytext=(0, 9), textcoords="offset points", ha="center", va="bottom", fontsize=10, color="#444444",
                    zorder=5, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.9})
    ax.set_ylim(0.45, 1.02)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("Median system-level NSE", fontsize=15)
    qr_dss, qr_ridge, area_ridge = [Line2D([], [], color=c, linestyle=ls, linewidth=2.0, marker=mk, markersize=6.5,
                                           markerfacecolor=c if f else "white", markeredgecolor=c) for _, c, ls, mk, f in SITE_LINES]
    square = Line2D([], [], linestyle="none", marker="s", markersize=7, color=AREA_GRAY)
    triangle = Line2D([], [], linestyle="none", marker="v", markersize=8, color=AREA_GRAY)
    ax.legend(handles=[qr_dss, (square, triangle), qr_ridge, area_ridge],
              labels=["QR sites, DSS decoder", "Area sites, DSS decoder", "QR sites, weighted ridge", "Area sites, weighted ridge"],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)}, **{"frameon": False, "fontsize": 10.5, "handlelength": 2.4, **legend})


def _paired(ax, effects: pd.DataFrame, items, ylim: tuple[float, float], legend_loc: str, sign: str) -> None:
    """Paired differences (mean over the 8 observed storms of the storm-median difference) with 95% bootstrap intervals; values above
    the axis are marked at the top with their values. `sign` (the legend title) states which side a positive difference favors."""
    handles = []
    for label, color, marker, offset, (first, second) in items:
        e = effects[(effects["first"] == first) & (effects["second"] == second)].set_index("m").loc[RANKS]
        y, lo, hi = e.mean_event_median_delta.to_numpy(), e.ci_lower.to_numpy(), e.ci_upper.to_numpy()
        x = np.asarray(RANKS, dtype=float) + offset
        inside = (hi <= ylim[1]) & (lo >= ylim[0])
        h = ax.errorbar(x[inside], y[inside], yerr=[y[inside] - lo[inside], hi[inside] - y[inside]], fmt=marker, color=color, ecolor=color,
                        elinewidth=1.35, capsize=4, markersize=math.sqrt(58.0), markeredgecolor="white", label=label, zorder=3)
        h[0].set_markeredgewidth(0.7)
        handles.append(h)
        for xi, yi in zip(x[~inside], y[~inside]):
            top = ylim[1] - 0.11 * (ylim[1] - ylim[0])
            ax.plot([xi], [top], marker="^", markersize=8.5, color=color, markeredgecolor="black", markeredgewidth=0.6, zorder=4)
            ax.annotate(f"+{yi:.3f}", (xi, top), xytext=(9, 0), textcoords="offset points", ha="left", va="center", fontsize=11, color="#4a4a4a",
                        zorder=5, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.9})
    ax.axhline(0.0, color="0.48", linewidth=0.75, linestyle="--", zorder=0)
    ax.set_ylim(*ylim)
    ax.legend(handles=handles, loc=legend_loc, frameon=False, fontsize=10.5, handlelength=1.0, handletextpad=0.4, title=sign,
              title_fontsize=10.5, alignment="left")


def figure7(table: pd.DataFrame, effects: pd.DataFrame) -> None:
    """Observed storms. (a) System NSE of QR, Uniform Random and the genetic algorithm; (b) median system NSE at the QR and
    largest-upstream-area sites with the DSS and event-weighted ridge decoders; (c) paired differences of placement (QR - GA-best with
    the DSS decoder; QR - Area with the weighted ridge decoder) and of decoder (DSS - weighted ridge at the QR sites)."""
    P.start(P.ARIAL)
    a_h, b_h, gap, top_in, bottom_in = 4.0, 4.15, 0.66, 0.12, 0.58
    height = top_in + a_h + gap + b_h + bottom_in
    fig = plt.figure(figsize=(12.8, height), dpi=600)
    grid = fig.add_gridspec(2, 2, height_ratios=[a_h, b_h], hspace=gap / ((a_h + b_h) / 2), wspace=0.21, left=0.07, right=0.985,
                            top=1 - top_in / height, bottom=bottom_in / height)
    ax = fig.add_subplot(grid[0, :])
    _placement_boxes(ax, table)
    _m_axis(ax, (0.5, 10.5))
    P.tag(ax, "a")
    ax = fig.add_subplot(grid[1, 0])
    _site_medians(ax, table, "observed", {"loc": "lower left", "bbox_to_anchor": (0.05, 0.15)})
    _m_axis(ax, (0.6, 10.4))
    P.tag(ax, "b")
    ax = fig.add_subplot(grid[1, 1])
    _paired(ax, effects, [("QR sites vs. GA-best sites (DSS decoder)", GA_PINK, "o", -0.22, ("QR_DSS", "GA_best_DSS")),
                          ("QR sites vs. Area sites (weighted ridge)", AREA_GRAY, "o", 0.0, ("QR_RidgeW", "Area_RidgeW")),
                          ("DSS decoder vs. weighted ridge (QR sites)", QR_BLUE, "s", 0.22, ("QR_DSS", "QR_RidgeW"))], (-0.062, 0.135), "upper right",
            "ΔNSE > 0: first option better")
    ax.set_ylabel("Paired ΔNSE", fontsize=15)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    _m_axis(ax, (0.55, 10.45))
    P.tag(ax, "c")
    P.save(fig, "Figure_7_placement", dpi=600, facecolor="white")
    plt.close(fig)


def figure_s2(ctx: dict, table: pd.DataFrame, effects: pd.DataFrame) -> dict[str, list[float]]:
    """(a) As Figure 7b for the 200-year design storm; (b) condition number of the sampled basis; (c) event-weighted minus unweighted
    ridge decoder at the QR sites, observed storms."""
    P.start(P.BOX_RC)
    width, height = 13.5, 4.7
    fig = plt.figure(figsize=(width, height), dpi=600)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.0], wspace=0.24, left=0.058, right=0.99, top=1 - 0.36 / height,
                            bottom=0.62 / height)
    axes = [fig.add_subplot(grid[0, k]) for k in range(3)]
    _site_medians(axes[0], table, "design", {"loc": "lower right", "bbox_to_anchor": (1.0, 0.16), "fontsize": 9.5, "handlelength": 2.0})
    axes[0].set_title("200-year design storm (n = 25)", fontsize=13, pad=6)
    ax = axes[1]
    conditions = {}
    for label, layout, color, marker in (("QR", "QR_DSS", QR_BLUE, "o"), ("GA-best", "GA_best", GA_PINK, "^"), ("Area", "Area", AREA_GRAY, "D")):
        conditions[layout] = [condition_number(ctx["basis"], ctx["layouts"][layout][m]) for m in RANKS]
        ax.plot(RANKS, conditions[layout], linestyle="-", linewidth=2, marker=marker, markersize=6.0, color=color, label=label)
    pd.DataFrame({"m": RANKS, **conditions}).to_csv(OUTPUTS_DIR / "Summary_condition_numbers.csv", index=False)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 2000)
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["1", "10", "100", "1000"])
    ax.set_ylabel("Condition number of sampled basis", fontsize=15)
    ax.legend(loc="upper left", frameon=False, fontsize=10.5, bbox_to_anchor=(0.06, 1.0), handletextpad=0.4)
    _paired(axes[2], effects, [("Weighted vs. unweighted ridge", WEIGHT_ORANGE, "o", 0.0, ("QR_RidgeW", "QR_Ridge"))], (-0.03, 0.31), "upper right",
            "ΔNSE > 0: weighted ridge better")
    axes[2].set_ylabel("Paired ΔNSE at the QR sites", fontsize=15)
    axes[2].set_title("Observed storms (n = 200)", fontsize=13, pad=6)
    for ax, text in zip(axes, "abc"):
        _m_axis(ax, (0.5, 10.5))
        P.letter(ax, text)
    P.save(fig, "Figure_S2_placement_details", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return conditions


def ridge_conduit_counts(ctx: dict) -> pd.DataFrame:
    """Observed storms, event-weighted ridge decoder, m = 5-10: active unmonitored conduits at the QR and Area sites, and how many
    of them have a per-conduit median NSE (over their active simulations) below 0.5."""
    observed = (ctx["scenarios"].rainfall_type == "observed").to_numpy()
    rows = []
    for sites in ("QR", "Area"):
        for m in range(5, 11):
            sensors, f = ridge(ctx, sites, m, weighted=True)
            metrics = conduit_metrics(ctx, sensors, f)
            act, nse = metrics["active"][observed], metrics["nse"][observed]
            active = [j for j in range(len(ctx["links"])) if j not in sensors and act[:, j].any()]
            below = [j for j in active if np.median(nse[act[:, j], j]) < 0.5]
            rows.append({"sites": sites, "m": m, "active_unmonitored": len(active), "median_nse_below_0p5": len(below)})
    counts = pd.DataFrame(rows)
    counts.to_csv(OUTPUTS_DIR / "Summary_ridge_conduit_level.csv", index=False)
    return counts
