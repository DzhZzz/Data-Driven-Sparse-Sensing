"""Check the regenerated outputs against the manuscript numerical contract."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from PIL import Image

from utils.paths import OUTPUTS_DIR


OUT = OUTPUTS_DIR


def close(actual: float, expected: float, tolerance: float = 1.0e-9) -> None:
    if not np.isfinite(actual) or abs(float(actual) - expected) > tolerance:
        raise RuntimeError(f"Numerical contract failed: {actual} != {expected}")


def run() -> None:
    selection = json.loads((OUT / "Figure_5_Selected_Events_and_Errors.json").read_text())
    roles = {str(item["size"]): str(item["catalog_id"]) for item in selection}
    if roles != {
        "Minimum depth": "OBS_20250608T0143",
        "Peak dominant": "OBS_20250723T1103",
        "Maximum depth": "OBS_20250715T1033",
    }:
        raise RuntimeError(f"Figure 5 event roles drifted: {roles}")

    figure6 = pd.read_csv(OUT / "Figure_6_Node_Median_Summary.csv")
    if int(figure6["Design_Node_Count"].max()) != 74:
        raise RuntimeError("Figure 6 must retain 74 evaluable design-event nodes.")

    figure7 = pd.read_csv(OUT / "Figure_7_Summary.csv")
    r8 = figure7.loc[figure7["r"].eq(8)].set_index("method")["median"]
    for method, expected in {
        "DSS-QR": 0.9698062243461072,
        "GA-Reconstruction": 0.9547733436912356,
        "Greedy D-opt": 0.9660131710732536,
        "Random placement": 0.3004375941706728,
    }.items():
        close(float(r8.loc[method]), expected)

    figure8 = pd.read_csv(OUT / "Figure_8_Noise_Summary.csv")
    r8 = figure8.loc[figure8["r"].eq(8)].set_index("scenario")["median"]
    for level, expected in {
        "clean": 0.9698062243461072,
        "5%": 0.9694663772802152,
        "10%": 0.9684445597564926,
        "15%": 0.9665123404369204,
    }.items():
        close(float(r8.loc[level]), expected)

    figure9 = pd.read_csv(OUT / "Figure_9_r8_sensor_means.csv")
    worst = figure9.loc[figure9["mean_signed_loss"].idxmax()]
    if str(worst["Node_ID"]) != "J304":
        raise RuntimeError("Figure 9 worst r=8 sensor must be J304.")
    close(float(worst["mean_signed_loss"]), 0.4259187516602237)

    associations = pd.read_csv(OUT / "Figure_10_Physical_Associations_r8.csv")
    expected_associations = {
        "Upstream drainage area": (77, 0.9770396951548304),
        "Mean circular-conduit diameter": (74, 0.9034003651405786),
        "Node activation fraction": (77, 0.8328962675416992),
        "Total node degree": (77, 0.0845889870016752),
    }
    for label, (count, rho) in expected_associations.items():
        row = associations.loc[associations["descriptor"].eq(label)].iloc[0]
        if int(row["n_nodes"]) != count:
            raise RuntimeError(f"Figure 10 node count drifted for {label}.")
        close(float(row["spearman_rho"]), rho)

    metrics = pd.read_csv(OUT / "Figure_10_Interpretive_Model_Metrics.csv").set_index("model")
    for label, expected in {
        "Upstream area": 0.3789305541355525,
        "Physical descriptors": 0.3755418415895299,
        "Modal exposure": 0.5424769696598442,
        "Physical plus modal exposure": 0.5589804925659485,
    }.items():
        close(float(metrics.loc[label, "leave_one_node_out_r2"]), expected)

    with Image.open(OUT / "Figure_10_Modal_Exposure_Failure_Risk_panel_a_scatter_alpha1.png") as image:
        if image.size != (5402, 4181):
            raise RuntimeError(f"Figure 10 dimensions drifted: {image.size}")
    print("PAPER_NUMERICAL_CONTRACT_OK")


if __name__ == "__main__":
    run()
