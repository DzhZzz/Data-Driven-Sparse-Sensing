# Data-driven sparse sensing of conduit flow in an urban drainage network

Code and data for the paper *Flow Reconstruction from Sparse Measurements in Urban Drainage Networks: A Simulation-Based Evaluation
of Data-Driven Sparse Sensing*.

## Run

From the repository root (Python 3.11 or newer):

```text
python -m pip install -r requirements.txt
python Code/reproduce_paper.py
```

The run takes about a minute. It writes Figures 3–8 and S2–S5, Tables S5–S6 and summary tables to `Outputs/`, checks the numbers
reported in the paper and ends with `PAPER_RESULTS_REPRODUCED`. Tested with Python 3.13, numpy 2.5, pandas 3.0, scipy 1.18 and
matplotlib 3.11.

## Contents

`Data/` holds the SWMM-simulated conduit flows. From them the code rebuilds the DSS basis and the QR layouts and computes the
reconstructions, decoders and metrics (`Code/utils/`: method; `Code/figures/`: figures; `Code/verify_results.py`: checked numbers).
Provided as results of the original analysis: the GA-best and largest-upstream-area layouts, the ridge penalties selected by cross-validation, Uniform Random
placement (median over the candidate layouts) and the measurement-error cases (median over 1,000 noise draws). Not included: the
SWMM model, the NSGA-II calibration, Figures 1, 2 and S1, and the location map of Figure 8b.

| File | Content |
|---|---|
| `development_flows.npz` | Flows (m³/s) of the 76 conduits for the 17 development storms (5,824 snapshots) |
| `heldout_flows.npz` | Flows of the 225 held-out simulations (25 parameter sets × 9 storms) |
| `frozen_basis.npz` | DSS basis of the paper, used to check the rebuilt basis |
| `sensor_layouts.csv` | QR (rebuilt and checked), GA-best and largest-upstream-area layouts, m = 1–10 |
| `ridge_penalties.csv` | Ridge penalties selected by cross-validation |
| `scenario_nse_random_placement.csv` | System NSE of Uniform Random placement per simulation |
| `scenario_nse_measurement_errors.npz` | System NSE of QR-DSS under measurement errors per simulation |
| `calibration_series.csv` | Rainfall and observed and simulated flow at OF-02 (Figure 3) |
| `rainfall_catalog.csv` | Development and held-out storms (Table S4) |
