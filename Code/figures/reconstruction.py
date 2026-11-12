"""Figures 4-6: system-level accuracy, held-out hydrographs and conduit-level accuracy of QR-DSS; Table S5."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator, PercentFormatter
from scipy.stats import gaussian_kde

from utils import data, plotting as P
from utils.core import decoder
from utils.evaluation import conduit_metrics, dss
from utils.paths import OUTPUTS_DIR

VALUE_LABEL = "conduit flow (m³/s)"
REAL, DESIGN = "#4a98c5", "#dc6d57"


# ------------------------------------------------------------------ Figure 4
def figure4(ctx: dict, system: pd.DataFrame) -> None:
    """(a) Median and Q10-Q90 of the system NSE (observed storms) and the design-storm median for m = 1-10;
    (b1-b3) simulated versus reconstructed conduit flow, pooled over the 225 simulations, at m = 1, 8 and 10."""
    P.start()
    ranks = list(data.RANKS)
    observed = {m: system[(system.m == m) & (system.rainfall_type == "observed")].system_nse.to_numpy() for m in ranks}
    design = {m: system[(system.m == m) & (system.rainfall_type == "design")].system_nse.to_numpy() for m in ranks}
    flows = ctx["flows"]
    hexbins = []
    for m in (1, 8, 10):
        sensors = ctx["layouts"]["QR_DSS"][m]
        hexbins.append({"label": f"m = {m}\nmedian NSE = {float(np.median(observed[m])):.3f}", "true_values": flows,
                        "pred_values": decoder(ctx["basis"], sensors, m) @ flows[sensors]})

    fig = plt.figure(figsize=(15, 10.1), dpi=300)
    outer = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], width_ratios=[1, 1, 1, 0.04], left=0.07, right=0.94, top=0.96,
                             bottom=0.09, hspace=0.195, wspace=0.10)
    ax = fig.add_subplot(outer[0, 0:3])
    bottom = [fig.add_subplot(outer[1, i]) for i in range(3)]
    cax = fig.add_subplot(outer[1, 3])
    size = 18
    x = np.asarray(ranks, dtype=float)
    per_m = lambda values, f: np.asarray([float(f(values[m])) for m in ranks])  # noqa: E731
    q10, q90 = per_m(observed, lambda v: np.quantile(v, 0.10)), per_m(observed, lambda v: np.quantile(v, 0.90))
    medians, design_medians = per_m(observed, np.median), per_m(design, np.median)
    ax.fill_between(x, q10, q90, color=REAL, alpha=0.18, linewidth=0, label="Observed storms, Q10–Q90")
    ax.plot(x, medians, color=REAL, linewidth=2.8, marker="o", markersize=6.0, label=f"Observed storms, median (n = {len(observed[1])})")
    ax.plot(x, design_medians, color=DESIGN, linewidth=2.8, marker="s", markersize=5.6,
            label=f"200-yr design storm, median (n = {len(design[1])})")
    pad = max(0.12, 0.02 * float(np.ptp(x)))
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_xticks(x)
    ax.set_xticklabels([int(v) for v in x])
    y_top, y_bottom = float(np.nanmax(np.concatenate([q90, design_medians]))), float(np.nanmin(np.concatenate([q10, design_medians])))
    y_lower = float(np.floor((y_bottom - 0.03) / 0.05) * 0.05)
    y_upper = y_top + max(0.02, 0.03 * (y_top - y_lower))
    ax.set_ylim(y_lower, y_upper)
    ax.yaxis.set_major_locator(MultipleLocator(0.1 if y_upper - y_lower <= 0.8 else 0.2))
    ax.set_xlabel("Number of monitored conduits, m", fontsize=size)
    ax.set_ylabel("System-level NSE", fontsize=size)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(axis="y", alpha=0.16, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#4a4a4a")
    ax.legend(fontsize=16, loc="lower right", frameon=False, facecolor="none")
    ax.text(0.015, 0.955, "a", transform=ax.transAxes, ha="left", va="top", fontsize=size, fontweight="bold")
    _hexbin_row(fig, bottom, cax, hexbins)
    P.save(fig, "Figure_4_system_level")
    plt.close(fig)


def _hexbin_row(fig, axes, cax, datasets) -> None:
    true = [np.asarray(d["true_values"]).reshape(-1) for d in datasets]
    pred = [np.asarray(d["pred_values"]).reshape(-1) for d in datasets]
    lo = min(min(np.min(v) for v in true), min(np.min(v) for v in pred))
    hi = max(max(np.max(v) for v in true), max(np.max(v) for v in pred))
    pad = (hi - lo) * 0.03 if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad
    last = None
    for k, (ax, d) in enumerate(zip(axes, datasets)):
        last = ax.hexbin(true[k], pred[k], gridsize=52, mincnt=1, cmap="RdBu_r", norm=LogNorm(), linewidths=0, extent=(lo, hi, lo, hi))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#2f2f2f", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_box_aspect(1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Simulated {VALUE_LABEL}", fontsize=18)
        if k == 0:
            ax.set_ylabel(f"Reconstructed {VALUE_LABEL}", fontsize=18)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, alpha=0.14, linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_edgecolor("#4a4a4a")
        if k > 0:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.text(0.045, 0.955, f"b{k + 1}", transform=ax.transAxes, ha="left", va="top", fontsize=16, fontweight="bold")
        ax.text(0.045, 0.875, d["label"], transform=ax.transAxes, ha="left", va="top", fontsize=16, linespacing=1.3)
    bar = fig.colorbar(last, cax=cax)
    bar.set_label("Counts", fontsize=16)
    bar.ax.tick_params(labelsize=16)
    for spine in bar.ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#666666")
    left, box = axes[0].get_position(), cax.get_position()
    cax.set_position([box.x0 - 0.006, left.y0, box.width * 1.18, left.height])


# ------------------------------------------------------------------ Figure 5
def figure5(ctx: dict, ranks=(1, 3, 8)) -> list[dict]:
    """Network-mean hydrographs of three observed held-out storms chosen without reconstruction results (minimum depth, largest
    ensemble-mean network peak, maximum depth). Each storm: mean over its 25 parameter sets, conduit by conduit; lines are
    network means and bands the 10th-90th percentiles across conduits. Returns the errors in the inset tables."""
    P.start()
    scenarios = ctx["scenarios"][ctx["scenarios"].rainfall_type == "observed"]
    catalog = data.rainfall_catalog()
    catalog = catalog[(catalog.Split.str.lower() == "testing") & (catalog.Event_Type.str.lower() == "observed")]
    catalog = catalog[catalog.Catalog_ID.isin(scenarios.catalog_id)].reset_index(drop=True)
    runs = {cid: rows.sort_values("parameter_set") for cid, rows in scenarios.groupby("catalog_id")}
    mean_flows = {cid: np.mean([ctx["flows"][:, r.start:r.stop] for r in rows.itertuples()], axis=0) for cid, rows in runs.items()}
    peak = catalog.Catalog_ID.map(lambda cid: float(mean_flows[cid].mean(axis=0).max()))
    chosen = [("Minimum-depth event", catalog.loc[catalog.Rain_Total_mm.idxmin()]), ("Peak-dominant event", catalog.loc[peak.idxmax()]),
              ("Maximum-depth event", catalog.loc[catalog.Rain_Total_mm.idxmax()])]
    panels = []
    for label, row in chosen:
        rows = runs[row.Catalog_ID]
        recon = {}
        for m in ranks:
            sensors = ctx["layouts"]["QR_DSS"][m]
            d = decoder(ctx["basis"], sensors, m)
            recon[m] = np.mean([d @ ctx["flows"][sensors, r.start:r.stop] for r in rows.itertuples()], axis=0)
        first = rows.iloc[0]
        stamps = pd.to_datetime(ctx["timestamps"][first.start:first.stop])
        panels.append({"observed": mean_flows[row.Catalog_ID], "reconstructed": recon,
                       "hours": (stamps - stamps[0]).total_seconds().to_numpy(dtype=float) / 3600.0,
                       "display_label": f"{label} | {str(row.Rain_Start_Local).split(' ')[0]} | total={float(row.Rain_Total_mm):.2f} mm"})
    return _hydrograph_figure(panels)


def _hydrograph_figure(panels: list[dict]) -> list[dict]:
    text_size = 16
    palette = [("#1f77b4", "#c6dbef", "s"), ("#2ca02c", "#c7e9c0", "^"), ("#d62728", "#fcbba1", "D")]
    minus = lambda text: text.replace("-", "−")  # noqa: E731
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 16 * len(panels) / 3), dpi=300, sharex=False)
    errors_by_panel = []
    for ax, panel, tag in zip(axes, panels, "abc"):
        observed, t = panel["observed"], panel["hours"]
        every = max(1, len(t) // 12)
        mean = observed.mean(axis=0)
        ax.plot(t, mean, linewidth=2.4, color="#6a51a3", marker="o", markersize=6.6, markevery=every, label="SWMM-simulated mean", zorder=4)
        ax.fill_between(t, np.nanpercentile(observed, 10, axis=0), np.nanpercentile(observed, 90, axis=0), color="#dadaeb", alpha=0.32,
                        edgecolor="none", linewidth=0, zorder=4)
        ranks = list(panel["reconstructed"])
        means = {}
        for k, m in enumerate(ranks):
            line, fill, marker = palette[k % len(palette)]
            values = panel["reconstructed"][m]
            means[m] = values.mean(axis=0)
            z = {0: 6, 1: 5}.get(k, 2)  # the smallest m on top, the largest beneath the SWMM line
            ax.plot(t, means[m], linewidth=2.0, color=line, marker=marker, markersize=6.2, markevery=every, label=f"m = {m} mean", zorder=z)
            ax.fill_between(t, np.nanpercentile(values, 10, axis=0), np.nanpercentile(values, 90, axis=0), color=fill, alpha=0.26,
                            edgecolor="none", linewidth=0, zorder=max(1, z - 1))
        i_peak = int(np.nanargmax(mean))
        volume = float(np.nansum(mean))
        errors = {m: {"peak": 100.0 * (float(np.nanmax(v)) / float(mean[i_peak]) - 1.0),
                      "dt_min": abs(float(t[int(np.nanargmax(v))]) - float(t[i_peak])) * 60.0,
                      "volume": 100.0 * (float(np.nansum(v)) / volume - 1.0)} for m, v in means.items()}
        errors_by_panel.append(errors)
        signed = lambda value: minus(f"{value:+.1f}" if abs(value) >= 1.0 else f"{value:+.2f}")  # noqa: E731
        cells = [[r"$E_{Q_{\mathrm{p}}}$ (%)"] + [signed(errors[m]["peak"]) for m in ranks],
                 [r"$|\Delta T_{\mathrm{p}}|$ (min)"] + [f"{errors[m]['dt_min']:.0f}" for m in ranks],
                 [r"$E_V$ (%)"] + [signed(errors[m]["volume"]) for m in ranks]]
        inset = ax.inset_axes([0.050, 0.625, 0.305, 0.235])
        inset.axis("off")
        table = inset.table(cellText=cells, colLabels=[""] + [f"$m={m}$" for m in ranks], cellLoc="center", rowLoc="center", colLoc="center",
                            bbox=[0.0, 0.0, 1.0, 1.0], colWidths=[0.35] + [0.65 / len(ranks)] * len(ranks))
        table.auto_set_font_size(False)
        table.set_fontsize(text_size - 4)
        table.scale(1.0, 1.05)
        for (r, c), cell in table.get_celld().items():
            cell.set_facecolor("white")
            cell.set_edgecolor("#b8b8b8")
            cell.set_linewidth(0.65)
            cell.PAD = 0.035
            if r == 0 and c > 0:
                cell.get_text().set_color(palette[(c - 1) % len(palette)][0])
                cell.get_text().set_weight("bold")
            if c == 0 and r > 0:
                cell.get_text().set_ha("left")
        ax.add_artist(AnchoredText(tag, loc="upper left", prop={"size": text_size, "weight": "bold"}, frameon=False))
        ax.add_artist(AnchoredText(panel["display_label"], loc="upper right", prop={"size": text_size}, frameon=False))
        step = 2.0 if float(t[-1]) <= 16.0 else 4.0  # 2-h ticks for short windows, 4-h ticks otherwise; none at the origin
        ticks = np.arange(step, max(step, float(t[-1])) + 1.0e-9, step)
        if tag == "a" and ticks.size:
            ticks = ticks[:-1]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: "{:02d}:{:02d}".format(*divmod(int(round(float(v) * 60.0)), 60))))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: "" if float(v) < -1.0e-10 else f"{float(v):.2f}"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))
        ax.set_xlim(0.0, float(t[-1]))
        ax.tick_params(axis="both", labelsize=text_size)
    fig.text(0.015, 0.5, f"Parameter-ensemble mean of network-mean {VALUE_LABEL}", va="center", rotation="vertical", fontsize=text_size)
    axes[-1].set_xlabel("Elapsed time (h)", fontsize=text_size, labelpad=6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0.035, 0.0, 1, 0.965])
    frame = axes[-1].get_position()
    fig.legend(handles, labels, fontsize=text_size, ncol=len(labels), frameon=False, loc="upper center",
               bbox_to_anchor=(frame.x0 + frame.width / 2.0, 0.985))
    P.save(fig, "Figure_5_hydrographs", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return errors_by_panel


# ------------------------------------------------------------------ Figure 6 and Table S5
def conduit_tables(ctx: dict) -> tuple[pd.DataFrame, dict]:
    """Per unmonitored conduit, rainfall type and m: active scenarios (simulated std >= 0.01 m³/s), median NSE over them, median NSE
    over all scenarios with a defined NSE, share of those with NSE < 0, median MAE and median temporal std. Also the m = 8 metrics."""
    observed = (ctx["scenarios"].rainfall_type == "observed").to_numpy()
    rows, metrics8 = [], None
    for m in data.RANKS:
        sensors = ctx["layouts"]["QR_DSS"][m]
        metrics = conduit_metrics(ctx, sensors, dss(ctx, sensors))
        if m == 8:
            metrics8 = metrics
        is_active = metrics["active"]
        for domain, mask in (("observed", observed), ("design", ~observed)):
            for j, link in enumerate(ctx["links"]):
                if j in metrics["sensors"]:
                    continue
                nse, act = metrics["nse"][mask, j], is_active[mask, j]
                defined = np.isfinite(nse)
                rows.append({"m": m, "domain": domain, "link_id": link, "n_active": int(act.sum()),
                             "median_nse_active": float(np.median(nse[act])) if act.any() else np.nan,
                             "n_defined": int(defined.sum()),
                             "median_nse_defined": float(np.median(nse[defined])) if defined.any() else np.nan,
                             "pct_defined_nse_below_0": 100.0 * float(np.mean(nse[defined] < 0)) if defined.any() else np.nan,
                             "median_mae_m3s": float(np.median(metrics["mae"][mask, j])),
                             "median_std_m3s": float(np.median(metrics["std"][mask, j]))})
    return pd.DataFrame(rows), metrics8


def table_s5(ctx: dict, metrics8: dict) -> pd.DataFrame:
    """Table S5: conduit-level results at m = 8 for activity thresholds of 0-0.02 m³/s (all 76 conduits)."""
    observed = (ctx["scenarios"].rainfall_type == "observed").to_numpy()
    rows = []
    for name, mask in (("Design", ~observed), ("Observed", observed)):
        nse, mae, std = metrics8["nse"][mask], metrics8["mae"][mask], metrics8["std"][mask]
        for threshold in (0.0, 0.001, 0.005, 0.01, 0.02):
            ok = (std >= threshold) & np.isfinite(nse)
            rows.append({"rainfall": name, "threshold_m3s": threshold, "evaluable_pairs": int(ok.sum()),
                         "median_evaluable_conduits": float(np.median(ok.sum(axis=1))), "median_nse": float(np.median(nse[ok])),
                         "q10_nse": float(np.quantile(nse[ok], 0.10)), "median_mae_Ls": 1000.0 * float(np.median(mae[ok]))})
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUTS_DIR / "Table_S5_activity_threshold.csv", index=False)
    return table


def figure6(conduits: pd.DataFrame) -> None:
    """(a) Per-conduit median NSE over active scenarios, unmonitored conduits, m = 1-10; (b) m = 8: median NSE over all scenarios
    with a defined NSE versus the share of them with NSE < 0; (c) m = 8: median absolute error versus median temporal std."""
    P.start(P.ARIAL)
    ranks = list(data.RANKS)
    classes = {"none": ("never active", "#9e9e9e"), "some": ("active in 1–100 scenarios", "#a6cee3"), "most": ("active in > 100 scenarios", REAL)}
    activity = lambda n: "none" if n == 0 else ("some" if n <= 100 else "most")  # noqa: E731
    fig = plt.figure(figsize=(12.8, 7.4), dpi=600)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.70, 1.0], hspace=0.20, wspace=0.15, left=0.075, right=0.985, top=0.95, bottom=0.085)

    ax = fig.add_subplot(grid[:, 0])
    medians = {(domain, m): conduits[(conduits.domain == domain) & (conduits.m == m) & (conduits.n_active > 0)].median_nse_active.to_numpy()
               for domain in ("observed", "design") for m in ranks}
    lower = min(-1.2, min(float(v.min()) for v in medians.values()) * 1.06)
    ax.set_yscale("function", functions=(P.forward, P.inverse))
    ax.set_ylim(lower, 1.02)
    ax.set_yticks(P.nse_ticks(lower))
    ax.yaxis.set_major_formatter(FuncFormatter(P.tick_label))
    spacing = 1.46
    box = {"widths": 0.28, "patch_artist": True, "zorder": 4, "whiskerprops": {"color": "black", "linewidth": 1.25},
           "capprops": {"color": "black", "linewidth": 1.25}, "medianprops": {"color": "black", "linewidth": 1.55}}
    for m in ranks:
        position = m * spacing
        for values, color, offset, side in ((medians[("observed", m)], REAL, -0.21, "left"), (medians[("design", m)], DESIGN, 0.21, "right")):
            _half_violin(ax, values, position + offset, color, 0.30, (lower, 1.02), side)
            ax.boxplot([values], positions=[position + offset], boxprops={"facecolor": color, "edgecolor": "black", "linewidth": 1.45, "alpha": 0.90},
                       flierprops={"marker": "o", "markersize": 2.4, "markerfacecolor": color, "markeredgecolor": color, "alpha": 0.55}, **box)
        ax.text(position, 1.012, f"{medians[('observed', m)].size} / {medians[('design', m)].size}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9.5, color="#555555")
    ax.text(spacing * 0.30, 1.012, "n =", transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=9.5, color="#555555")
    ax.set_xticks([m * spacing for m in ranks])
    ax.set_xticklabels([str(m) for m in ranks])
    ax.set_xlim(spacing * 0.30, spacing * 10.45)
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.9, zorder=0)
    ax.set_xlabel("Number of monitored conduits, m", fontsize=15)
    ax.set_ylabel("Conduit-level median NSE", fontsize=15)
    ax.grid(axis="y", alpha=0.15, linewidth=0.75)
    ax.legend([plt.Rectangle((0, 0), 1, 1, facecolor=REAL, edgecolor="black", linewidth=1.2),
               plt.Rectangle((0, 0), 1, 1, facecolor=DESIGN, edgecolor="black", linewidth=1.2)], ["Observed storms", "200-year design storm"],
              loc="lower right", bbox_to_anchor=(0.98, 0.01), frameon=False, fontsize=11.0, handletextpad=0.35)
    P.tag(ax, "a")
    P.style(ax)

    m8 = conduits[(conduits.domain == "observed") & (conduits.m == 8)].copy()
    m8["activity"] = [activity(n) for n in m8.n_active]
    b = m8[m8.n_defined > 0]
    ax = fig.add_subplot(grid[0, 1])
    x_lower = min(-1.2, float(b.median_nse_defined.min()) * 1.15)
    ax.set_xscale("function", functions=(P.forward, P.inverse))
    ax.set_xlim(x_lower, 1.06)
    ticks = P.nse_ticks(x_lower)
    ax.set_xticks([v for v in ticks if v in (-20.0, -5.0, -1.0, 0.0, 0.5, 1.0) or v == min(ticks)])
    ax.xaxis.set_major_formatter(FuncFormatter(P.tick_label))
    for key, (_, color) in classes.items():
        rows = b[b.activity == key]
        ax.scatter(rows.median_nse_defined, rows.pct_defined_nse_below_0 / 100.0, s=34, marker="o", facecolor=color, edgecolor="white",
                   linewidth=0.55, alpha=0.88, zorder=3 if key == "none" else 4)
    ax.axvline(0.5, color="#555555", linestyle="--", linewidth=1.0, zorder=1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(-0.05, 1.18)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Conduit median NSE, m = 8", fontsize=13)
    ax.set_ylabel("Negative-NSE occurrence rate", fontsize=13)
    ax.grid(alpha=0.15, linewidth=0.75)
    P.tag(ax, "b")
    P.style(ax, 11.5)

    ax = fig.add_subplot(grid[1, 1])
    floor = 0.012
    std_lps = m8.median_std_m3s * 1000.0
    for key, (_, color) in classes.items():
        rows = m8[(m8.activity == key) & (std_lps > 0.0)]
        ax.scatter(rows.median_std_m3s * 1000.0, rows.median_mae_m3s * 1000.0, s=30, marker="o", facecolor=color, edgecolor="white",
                   linewidth=0.55, alpha=0.88, zorder=4)
    zero = m8[std_lps <= 0.0]
    ax.scatter(np.full(len(zero), floor), zero.median_mae_m3s * 1000.0, s=30, marker="o", facecolor="white", edgecolor="#777777",
               linewidth=1.0, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.009, 250)
    ax.set_ylim(0.008, 6)
    ax.plot([0.009, 6], [0.009, 6], color="#555555", linestyle="--", linewidth=1.0, zorder=1)
    ax.axvline(10.0, color="#555555", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(11.5, 5.2, "10 L/s activity\nthreshold", fontsize=8.5, color="#555555", ha="left", va="top", linespacing=1.1)
    p0, p1 = ax.transData.transform([(0.1, 0.1), (1.0, 1.0)])  # label parallel to the 1:1 line on screen
    angle = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
    ax.annotate("MAE = std", xy=(0.9, 0.9), xytext=(-3.0 * np.sin(np.radians(angle)), 3.0 * np.cos(np.radians(angle))), textcoords="offset points",
                rotation=angle, rotation_mode="anchor", ha="center", va="bottom", fontsize=9.0, color="#555555")
    ax.set_xlabel("Median temporal std of the simulated flow (L/s)", fontsize=13)
    ax.set_ylabel("Median absolute error (L/s)", fontsize=13)
    ax.grid(alpha=0.15, linewidth=0.75)
    short = {"none": "none", "some": "1–100", "most": "> 100"}
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=6.0, markerfacecolor=color, markeredgecolor="white", label=short[key])
               for key, (_, color) in classes.items()]
    handles.append(Line2D([], [], marker="o", linestyle="none", markersize=6.0, markerfacecolor="white", markeredgecolor="#777777", label="zero flow"))
    ax.legend(handles=handles, title="Active scenarios\n(of 200)", title_fontsize=8.5, loc="lower right", frameon=False, fontsize=8.5,
              handletextpad=0.1, borderaxespad=0.25, labelspacing=0.25, alignment="left")
    P.tag(ax, "c")
    P.style(ax, 11.5)
    P.save(fig, "Figure_6_conduit_level", dpi=600, facecolor="white")
    plt.close(fig)


def _half_violin(ax, values, position, color, width, limits, side) -> None:
    values = values[np.isfinite(values)]
    if values.size < 2 or np.nanstd(values) == 0.0:
        return
    y_min, y_max = max(limits[0], values.min() - 0.08), min(limits[1], values.max() + 0.08)
    grid = np.linspace(float(P.forward(y_min)), float(P.forward(y_max)), 400)
    density = gaussian_kde(P.forward(values))(grid)
    scaled = density / density.max() * width * (-1.0 if side == "left" else 1.0)
    ax.fill_betweenx(P.inverse(grid), position, position + scaled, facecolor=color, edgecolor="none", linewidth=0.0, alpha=0.52, zorder=4)
