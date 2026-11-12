"""Check the data and the reproduced results against the numbers printed in the paper (main text, Text S6, Tables S5-S6).

Each number must round to the value printed in the paper (half a unit of its last printed digit).
"""

from __future__ import annotations

import numpy as np

from utils import data

MS = list(data.RANKS)


def run(r: dict) -> None:
    checks = []

    def close(label, value, printed, decimals):
        checks.append((label, float(value), printed, abs(float(value) - printed) <= 0.5 * 10.0 ** -decimals + 1e-9))

    def true(label, condition, value=None):
        checks.append((label, value, None, bool(condition)))

    t = r["table"]
    grouped = t.groupby(["method", "rainfall_type", "m"]).system_nse
    med, q10 = grouped.median(), grouped.quantile(0.1)
    eff = r["effects"].set_index(["first", "second", "m"]).sort_index()
    loss = r["losses"].set_index(["variant", "m"]).sort_index()
    con = r["conduits"]

    # Section 2.3 and Figure 3: calibration (E01, E04) and validation (E02) at OF-02
    rain = data.calibration_series().groupby("Event_ID").Rain_SWMM_Primary_mm.sum()
    for event, depth in (("E01", 11.33), ("E02", 6.34), ("E04", 7.27)):
        close(f"rainfall depth {event} (mm)", rain[event], depth, 2)
    for event, value in (("E01", 0.559), ("E04", 0.553), ("E02", 0.612)):
        close(f"NSE {event}", r["calibration"][event]["nse"], value, 3)
    close("E02 peak bias %", r["calibration"]["E02"]["peak_bias_pct"], -46.1, 1)
    close("E02 volume bias %", r["calibration"]["E02"]["volume_bias_pct"], -42.1, 1)
    # Section 2.4 and Text S6: development storms, basis weighting by storm-wise cross-validation, spectrum
    dev, state, events, offsets = data.development()
    design_storm = np.array([e.startswith("DESIGN") for e in events])
    energy = np.array([np.sum(dev[:, a:b] ** 2) for a, b in zip(offsets[:-1], offsets[1:])])
    true("76 conduits, 5,824 snapshots of 17 storms (5 design)", (dev.shape, len(events), int(design_storm.sum())) == ((76, 5824), 17, 5))
    close("design storms, share of the squared development flow %", 100.0 * energy[design_storm].sum() / energy.sum(), 92.0, 1)
    close("design storms, share of the development snapshots %", 100.0 * np.diff(offsets)[design_storm].sum() / offsets[-1], 37.1, 1)
    close("CV criterion, event-energy weighting", r["cv"]["event_energy_weighted"], 0.954, 3)
    close("CV criterion, snapshot-equal weighting", r["cv"]["snapshot_equal"], 0.842, 3)
    for k, value in ((1, 82.879), (2, 95.778), (5, 99.219), (8, 99.706)):
        close(f"cumulative energy % at r = {k}", r["cumulative_energy"][k - 1], value, 3)
    # Section 2.5: held-out simulations
    held = t[(t.method == "QR_DSS") & (t.m == 1)]
    true("225 held-out simulations: 9 storms x 25 parameter sets, 200 observed",
         held.catalog_id.nunique() == 9 and held.groupby("catalog_id").size().eq(25).all() and int((held.rainfall_type == "observed").sum()) == 200)
    # Section 3.1 and Figure 4: system-level NSE of QR-DSS
    for m, mm, qq, dd in ((1, 0.855, 0.828, 0.598), (3, 0.934, 0.876, 0.961), (5, 0.992, 0.973, 0.977), (8, 0.996, 0.982, 0.988)):
        close(f"QR-DSS observed median, m = {m}", med[("QR_DSS", "observed", m)], mm, 3)
        close(f"QR-DSS observed Q10, m = {m}", q10[("QR_DSS", "observed", m)], qq, 3)
        close(f"QR-DSS design median, m = {m}", med[("QR_DSS", "design", m)], dd, 3)
    qr_obs = t[(t.method == "QR_DSS") & (t.rainfall_type == "observed")]
    for m, low, high in ((3, 0.877, 0.964), (5, 0.972, 0.997)):
        storms = qr_obs[qr_obs.m == m].groupby("catalog_id").system_nse.median()
        close(f"storm medians, m = {m}, lowest", storms.min(), low, 3)
        close(f"storm medians, m = {m}, highest", storms.max(), high, 3)
    close("event-weighted minus snapshot-equal basis, m = 3", eff.loc[("QR_DSS", "QR_DSS_snapshot_equal", 3), "mean_event_median_delta"], 0.078, 3)
    close("event-weighted minus snapshot-equal basis, m = 8", eff.loc[("QR_DSS", "QR_DSS_snapshot_equal", 8), "mean_event_median_delta"], 0.107, 3)
    # Figure 5: errors of the network-mean hydrographs over the three storms
    errors = r["hydrographs"]
    for m, key, low, high, d_low, d_high in ((1, "peak", 52.0, 69.7, 1, 1), (3, "peak", 4.1, 32.8, 1, 1), (8, "peak", 0.05, 4.0, 2, 1),
                                             (1, "volume", 32.6, 39.1, 1, 1), (3, "volume", 7.4, 14.6, 1, 1), (8, "volume", 0.45, 1.4, 2, 1)):
        values = [abs(panel[m][key]) for panel in errors]
        close(f"Figure 5 |{key} error| %, m = {m}, lowest", min(values), low, d_low)
        close(f"Figure 5 |{key} error| %, m = {m}, highest", max(values), high, d_high)
    true("Figure 5 peak-time differences within one 5-min step", max(p[m]["dt_min"] for p in errors for m in p) <= 5.0 + 1e-6)
    # Section 3.2 and Figure 6: conduit level, observed storms
    obs, design = con[con.domain == "observed"], con[con.domain == "design"]
    active = lambda frame, m: frame[(frame.m == m) & (frame.n_active > 0)]  # noqa: E731
    for m, mm, mn, n, n_active in ((3, 0.492, -6.21, 73, 23), (5, 0.967, 0.681, 71, 21), (8, 0.948, 0.798, 68, 18)):  # also Table S6
        a = active(obs, m).median_nse_active
        close(f"active conduits, median of per-conduit medians, m = {m}", a.median(), mm, 3)
        close(f"active conduits, lowest per-conduit median, m = {m}", a.min(), mn, 2 if m == 3 else 3)
        true(f"m = {m}: {n_active} of the {n} unmonitored conduits active", (int((obs.m == m).sum()), len(a)) == (n, n_active))
    m8 = obs[obs.m == 8]
    below = m8[m8.median_nse_defined < 0.5]
    true("m = 8: 33 conduits with median NSE below 0.5, none ever active", len(below) == 33 and (below.n_active == 0).all(), len(below))
    close("m = 8: largest median absolute error (L/s)", 1000.0 * m8.median_mae_m3s.max(), 1.9, 1)
    close("design storm, m = 8, median of per-conduit medians", active(design, 8).median_nse_active.median(), 0.880, 3)
    n_design = [len(active(design, m)) for m in MS]
    true("design storm: 58-67 active unmonitored conduits", (min(n_design), max(n_design)) == (58, 67), n_design)
    only = [len(set(active(design, m).link_id) - set(active(obs, m).link_id)) for m in MS]
    true("design storm: 41-42 conduits active only under it", (min(only), max(only)) == (41, 42), only)
    true("design storm: lower median than the observed storms for m >= 5",
         all(active(design, m).median_nse_active.median() < active(obs, m).median_nse_active.median() for m in range(5, 11)))
    both = [active(design, m)[active(design, m).link_id.isin(active(obs, m).link_id)].median_nse_active.median() for m in range(5, 11)]
    close("design storm, m >= 5, conduits also active in the observed storms, lowest median", min(both), 0.97, 2)
    close("design storm, m >= 5, conduits also active in the observed storms, highest median", max(both), 0.99, 2)
    # Section 3.3, Figure 7 and Figure S2: placement and decoders
    random = [med[("Random_median", "observed", m)] for m in MS]
    close("Uniform Random medians, lowest", min(random), 0.09, 2)
    close("Uniform Random medians, highest", max(random), 0.54, 2)
    ga = eff.loc[("QR_DSS", "GA_best_DSS")]
    true("GA-best exceeds QR at m = 1, 3, 4, 7 and 10", list(ga.index[ga.ci_upper < 0]) == [1, 3, 4, 7, 10])
    true("QR exceeds GA-best at m = 8 only", list(ga.index[ga.ci_lower > 0]) == [8])
    true("all QR - GA-best differences at most 0.016", ga.mean_event_median_delta.abs().max() <= 0.016 + 5e-4)
    for column, value in (("mean_event_median_delta", -0.0155), ("ci_lower", -0.0230), ("ci_upper", -0.0082)):
        close(f"QR - GA-best, m = 3, {column}", ga.loc[3, column], value, 4)
    close("Area sites, DSS decoder, observed median, m = 3", med[("Area_DSS", "observed", 3)], -16.7, 1)
    close("condition number, Area, m = 10", r["conditions"]["Area"][9], 459, 0)
    close("condition number, QR, m = 10", r["conditions"]["QR_DSS"][9], 4.6, 1)
    area = eff.loc[("QR_RidgeW", "Area_RidgeW")]
    for m, value in ((1, 0.113), (8, 0.012), (3, 0.013)):
        close(f"QR - Area sites, weighted ridge, m = {m}", area.loc[m, "mean_event_median_delta"], value, 3)
    close("QR - Area sites, weighted ridge, m = 3, lower", area.loc[3, "ci_lower"], -0.003, 3)
    close("QR - Area sites, weighted ridge, m = 3, upper", area.loc[3, "ci_upper"], 0.035, 3)
    true("QR ahead of the Area sites (weighted ridge) at m = 1 and m >= 4 only", list(area.index[area.ci_lower > 0]) == [1, 4, 5, 6, 7, 8, 9, 10])
    rc = r["ridge_conduits"]
    a_rc, q_rc = rc[rc.sites == "Area"], rc[rc.sites == "QR"]
    true("Area sites, weighted ridge, m >= 5: 7-12 of 16-21 active conduits below 0.5",
         (a_rc.median_nse_below_0p5.min(), a_rc.median_nse_below_0p5.max(), a_rc.active_unmonitored.min(), a_rc.active_unmonitored.max()) == (7, 12, 16, 21))
    true("QR sites, weighted ridge, m >= 5: no active conduit below 0.5", (q_rc.median_nse_below_0p5 == 0).all())
    dec = eff.loc[("QR_DSS", "QR_RidgeW")]
    close("DSS - weighted ridge at QR, m = 8", dec.loc[8, "mean_event_median_delta"], -0.0009, 4)
    for column, value in (("mean_event_median_delta", -0.031), ("ci_lower", -0.039), ("ci_upper", -0.023)):
        close(f"DSS - weighted ridge at QR, m = 3, {column}", dec.loc[3, column], value, 3)
    for m, value in ((3, 0.139), (8, 0.014)):
        close(f"weighted - unweighted ridge at QR, m = {m}", eff.loc[("QR_RidgeW", "QR_Ridge", m), "mean_event_median_delta"], value, 3)
    for method, low, high in (("Area_RidgeW", 0.64, 0.68), ("QR_RidgeW", 0.92, 0.96)):
        d = [med[(method, "design", m)] for m in (2, 3, 4)]
        close(f"{method}, design storm, m = 2-4, lowest", min(d), low, 2)
        close(f"{method}, design storm, m = 2-4, highest", max(d), high, 2)
    # Section 3.4, Figure 8a and Figure S3: measurement errors, m = 8
    errors_ = data.measurement_errors()
    observed = errors_["rainfall_types"] == "observed"
    close("m = 8, error-free", med[("QR_DSS", "observed", 8)], 0.996, 3)
    for key, value in (("5pct", 0.995), ("10pct", 0.994), ("15pct", 0.993), ("detection_floor_10Ls", 0.990), ("bias_all_minus20pct", 0.957),
                       ("bias_all_plus20pct", 0.954), ("additive_2Ls", 0.989), ("additive_5Ls", 0.973), ("additive_10Ls", 0.916)):
        close(f"m = 8, {key}", np.median(errors_[f"{key}_m08"][observed]), value, 3)
    close("m = 8, additive 10 L/s, design storm", np.median(errors_["additive_10Ls_m08"][~observed]), 0.988, 3)
    # Section 3.5, Figure 8b-c: loss of one monitored conduit
    per8 = r["per_sensor_m8"]
    true("m = 8: six of the eight losses below 0.08", sum(v < 0.08 for v in per8.values()) == 6)
    close("m = 8, loss of Link-686", per8["Link-686"], 0.419, 3)
    close("m = 8, loss of Link-682", per8["Link-682"], 0.393, 3)
    for variant, m, mean, worst in (("uncompensated", 2, 0.416, 0.772), ("uncompensated", 8, 0.125, 0.419), ("uncompensated", 10, 0.084, 0.402),
                                    ("refit", 8, 0.003, 0.012)):
        close(f"{variant}, m = {m}, mean loss", loss.loc[(variant, m), "mean_median_loss"], mean, 3)
        close(f"{variant}, m = {m}, worst loss", loss.loc[(variant, m), "worst_median_loss"], worst, 3)
    close("refit, worst loss for m >= 3 (largest)", max(loss.loc[("refit", m), "worst_median_loss"] for m in range(3, 11)), 0.071, 3)
    truncated, uncompensated = loss.loc["truncated"].mean_median_loss, loss.loc["uncompensated"].mean_median_loss
    true("truncated basis: lower mean loss than uncompensated at some m, higher at others", (truncated < uncompensated).any() and (truncated > uncompensated).any())
    # Table S5: activity-threshold sensitivity at m = 8
    s5 = r["table_s5"]
    printed = [(1900, 76, 0.894, 0.489, 5.894), (1900, 76, 0.894, 0.489, 5.894), (1816, 73, 0.898, 0.483, 6.319), (1686, 68, 0.897, 0.481, 6.963),
               (1511, 61, 0.910, 0.533, 8.670), (14400, 72, 0.776, -18.845, 0.138), (7388, 38, 0.946, 0.670, 0.386), (4018, 21, 0.994, 0.877, 0.542),
               (2769, 13, 0.999, 0.928, 0.429), (2250, 13, 1.000, 0.946, 0.179)]
    for row, (pairs, n, nse, q, mae) in zip(s5.itertuples(), printed):
        label = f"Table S5, {row.rainfall}, {row.threshold_m3s:.3f} m3/s"
        true(f"{label}, evaluable pairs and conduits", (row.evaluable_pairs, row.median_evaluable_conduits) == (pairs, n))
        close(f"{label}, median NSE", row.median_nse, nse, 3)
        close(f"{label}, Q10 NSE", row.q10_nse, q, 3)
        close(f"{label}, median MAE (L/s)", row.median_mae_Ls, mae, 3)
    # Table S6: monitoring-design summary
    printed = {3: (0.934, 0.876, 0.961, 0.492, -6.21, 0.59, 10.3, 0.930, 0.914, 0.255, 0.678, 0.005, 0.010),
               5: (0.992, 0.973, 0.977, 0.967, 0.681, 0.16, 3.7, 0.990, 0.965, 0.241, 0.656, 0.027, 0.063),
               8: (0.996, 0.982, 0.988, 0.948, 0.798, 0.15, 1.9, 0.993, 0.973, 0.125, 0.419, 0.003, 0.012)}
    columns = ["system_nse_median", "system_nse_q10", "design_median", "active_conduit_median_nse_median", "active_conduit_median_nse_min",
               "conduit_mae_median_Ls", "conduit_mae_max_Ls", "multiplicative_15pct_median", "additive_5Ls_median", "uncompensated_mean",
               "uncompensated_worst", "refit_mean", "refit_worst"]
    decimals = [3, 3, 3, 3, None, 2, 1, 3, 3, 3, 3, 3, 3]
    for row in r["table_s6"].itertuples():
        for column, value, d in zip(columns, printed[row.m], decimals):
            close(f"Table S6, m = {row.m}, {column}", getattr(row, column), value, d if d is not None else (2 if row.m == 3 else 3))

    for label, value, printed_value, ok in checks:
        shown = "" if value is None else (f" = {value:.6g}" if isinstance(value, float) else f" = {value}") + ("" if printed_value is None else f" (paper {printed_value})")
        print(f"{'ok  ' if ok else 'FAIL'} {label}{shown}")
    failed = sum(not ok for *_, ok in checks)
    if failed:
        raise SystemExit(f"{failed} of {len(checks)} checks of the paper's numbers failed")
    print(f"All {len(checks)} checks of the paper's numbers passed.")
