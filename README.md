# Data-driven sparse sensing for urban drainage flow reconstruction

This repository contains the compact Python code and derived data needed to
reproduce manuscript Figures 3-10. The EPA-SWMM model and all SWMM input,
report, and binary output files are excluded.

## Run

Use Python 3.11 or newer from the repository root:

```text
python -m pip install -r requirements.txt
python Code/reproduce_paper.py
```

The command creates `Outputs/`, regenerates the figures and numerical tables,
checks the paper's numerical contract, and ends with:

```text
PAPER_RESULTS_REPRODUCED
```

## Included data

- `development_flows.npz`: 17 nominal development storms, 77 nodes, and 5,824
  five-minute snapshots used for the uncentered snapshot-equal direct SVD.
- `heldout_flows.npz`: 225 evaluation simulations formed by 25 plausible
  calibrated parameter sets crossed with nine rainfall events.
- `frozen_basis.npz` and `sensor_layouts.csv`: the fixed direct-SVD basis,
  normalization scale, and QR sensor sets used in the paper.
- Compact scenario-level plotting data for the placement, noise, sensor-loss,
  modal-exposure, and physical-grounding results.
- Calibration event series and the two sensor-layout map panels used by the
  manuscript figures.

Held-out flows are used only for evaluation. They do not enter scaling, basis
construction, sensor-layout selection, or decoder definition.

## Code map

The root of `Code/` contains only the two public entry points:

- `reproduce_paper.py`: regenerates every included manuscript result.
- `verify_results.py`: checks the regenerated files against the manuscript's
  numerical contract.

Reusable implementation is grouped by responsibility:

- `utils/core.py`: direct SVD, pivoted-QR sensor selection, and reconstruction.
- `utils/data.py`: compact-data loading and data-contract validation.
- `utils/metrics.py`: the formal NSE definitions used by the paper.
- `utils/reconstruction.py`: held-out reconstruction and node/system metrics.
- `utils/plotting.py`: plotting helpers shared by Figures 4-6.
- `utils/paths.py`: repository-relative data and output paths.
- `figures/`: one focused module for each figure group: calibration (Figure 3),
  reconstruction (Figures 4-6), placement (Figure 7), and robustness
  (Figures 8-10).

The compact Figure 8 data contain the 225 scenario-level values actually
plotted after reducing 1,000 perturbation draws within each scenario. The
compact Figure 9 data retain the signed sensor-loss matrix needed by the
published plot, without duplicated baseline and dropout matrices.
