# DSS
Optimizing Sensor Placement for Flow Reconstruction in Urban Drainage Networks: A Digital Twin-Based Sparse Sensing Approach

## Private SWMM model configuration

This repository keeps SWMM-derived datasets, cached analysis files, and result tables/figures for reproducibility. The private SWMM `.inp` model itself is not included.

Before running menu options that call SWMM directly, provide your own local model path:

```powershell
$env:DSS_SWMM_MODEL_PATH = "C:\path\to\your_model.inp"
```

If `DSS_SWMM_MODEL_PATH` is not set, SWMM menu options will prompt for a local `.inp` file at runtime.

The default training/testing CSV paths remain under `Training/` and `Testing/` so the published analysis can be checked without the protected SWMM model.
