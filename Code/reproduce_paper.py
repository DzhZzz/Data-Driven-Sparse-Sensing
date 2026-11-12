"""Reproduce Figures 3-8 and S2-S5 and check the numbers of the paper, from the derived data in Data/.

Run from the repository root:  python Code/reproduce_paper.py
Outputs (figures, Tables S5-S6 and summary tables) are written to Outputs/; the run ends with PAPER_RESULTS_REPRODUCED.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import verify_results  # noqa: E402
from figures import calibration, placement, reconstruction, robustness, spectrum  # noqa: E402
from utils.evaluation import basis_weighting_cv, context, sensor_loss  # noqa: E402
from utils.paths import OUTPUTS_DIR  # noqa: E402


def table_s6(table, conduits, values, losses) -> pd.DataFrame:
    """Monitoring-design summary for m = 3, 5 and 8 (observed storms unless stated)."""
    rows = []
    for m in (3, 5, 8):
        qr = table[(table.method == "QR_DSS") & (table.m == m)]
        observed = qr[qr.rainfall_type == "observed"].system_nse
        unmonitored = conduits[(conduits.domain == "observed") & (conduits.m == m)]
        active = unmonitored[unmonitored.n_active > 0].median_nse_active
        rows.append({"m": m, "share_of_76_pct": 100.0 * m / 76, "system_nse_median": observed.median(), "system_nse_q10": observed.quantile(0.1),
                     "design_median": qr[qr.rainfall_type == "design"].system_nse.median(),
                     "active_conduit_median_nse_median": active.median(), "active_conduit_median_nse_min": active.min(),
                     "conduit_mae_median_Ls": 1000.0 * unmonitored.median_mae_m3s.median(), "conduit_mae_max_Ls": 1000.0 * unmonitored.median_mae_m3s.max(),
                     "multiplicative_15pct_median": float(np.median(values[f"15pct_m{m:02d}"])),
                     "additive_5Ls_median": float(np.median(values[f"additive_5Ls_m{m:02d}"])),
                     **{f"{v}_{s}": losses.set_index(["variant", "m"]).loc[(v, m), f"{s}_median_loss"]
                        for v in ("uncompensated", "refit") for s in ("mean", "worst")}})
    table6 = pd.DataFrame(rows)
    table6.to_csv(OUTPUTS_DIR / "Table_S6_monitoring_design.csv", index=False)
    return table6


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUTS_DIR.iterdir():
        path.unlink()
    ctx = context()  # basis and QR layouts rebuilt from the development storms and checked
    r = {"calibration": calibration.figure3(), "cv": basis_weighting_cv(ctx)}
    r["table"] = table = placement.scenario_table(ctx)
    system = table[table.method == "QR_DSS"]
    reconstruction.figure4(ctx, system)
    r["hydrographs"] = reconstruction.figure5(ctx)
    r["conduits"], metrics8 = reconstruction.conduit_tables(ctx)
    reconstruction.figure6(r["conduits"])
    r["table_s5"] = reconstruction.table_s5(ctx, metrics8)
    r["ridge_conduits"] = placement.ridge_conduit_counts(ctx)
    r["effects"] = placement.paired_effects(table)
    placement.figure7(table, r["effects"])
    r["conditions"] = placement.figure_s2(ctx, table, r["effects"])
    values = robustness.error_values(system)
    losses = robustness.loss_values(sensor_loss(ctx))
    r["losses"] = robustness.loss_summary(ctx, losses)
    r["per_sensor_m8"] = {link: float(np.median(losses[("uncompensated", 8, link)])) for (_, m, link) in losses if m == 8}
    robustness.figure8(ctx, values, losses, r["losses"])
    robustness.figure_s3(values)
    robustness.figure_s4(ctx, losses)
    r["cumulative_energy"] = spectrum.figure_s5(ctx)
    r["table_s6"] = table_s6(table, r["conduits"], values, r["losses"])
    verify_results.run(r)
    print("PAPER_RESULTS_REPRODUCED")


if __name__ == "__main__":
    main()
