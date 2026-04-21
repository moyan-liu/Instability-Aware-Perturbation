# FTLE-Guided Atmospheric River Perturbation

Research code for identifying optimal cloud seeding locations to reduce landfalling Atmospheric River (AR) precipitation using the [Microsoft Aurora](https://github.com/microsoft/aurora) weather model and Finite-Time Lyapunov Exponents (FTLE).

**Best result:** 5.21% IVT reduction at the California coast; site at 30.8°N, 166.5°E (2022-12-24 12Z AR event, 72–84h forecast).

---

## Repository Structure

```
aurora-ar-perturbation/
├── notebooks/
│   └── perturb_paper.ipynb          # Main pipeline notebook
├── packages/
│   ├── aurora-seeding/              # Physically consistent cloud seeding perturbations
│   ├── ftle-perturbation/           # FTLE-guided candidate scoring & rollout
│   └── ftle_calculation/            # GPU-accelerated FTLE computation
├── scripts/
│   ├── ca_boundary.py               # California coastal boundary utilities
│   ├── ca_pixel_map.py              # CA pixel grid mapping
│   ├── plot_single_timestep.py      # Standalone timestep visualization
│   └── cloud_seeding_perturbation_logic.md  # Physics notes
└── README.md
```

---

## Setup

### Requirements

- Python >= 3.10
- CUDA-capable GPU (Aurora inference requires GPU)
- [Microsoft Aurora](https://github.com/microsoft/aurora): `pip install microsoft-aurora`
- ERA5 data (see Data section below)

### Install the custom packages

```bash
pip install -e packages/aurora-seeding
pip install -e packages/ftle-perturbation
pip install -e packages/ftle_calculation
```

### Install PyPI dependencies

```bash
pip install torch numpy xarray pandas matplotlib cartopy scipy
```

---

## Data

ERA5 reanalysis data is required but **not included** in this repository (too large).

You need daily NetCDF files in a directory (e.g. `data/era5_batch/`):

| File | Variables |
|---|---|
| `static.nc` | `z` (orography), `slt` (soil type), `lsm` (land-sea mask) |
| `YYYY-MM-DD-surface-level.nc` | `t2m`, `u10`, `v10`, `msl` |
| `YYYY-MM-DD-atmospheric.nc` | `t`, `u`, `v`, `q`, `z` on pressure levels |

Download via the [CDS API](https://cds.climate.copernicus.eu/) at 0.25° resolution. For the 2022-12-24 AR case study, you need data from 2022-12-20 to 2023-01-15.

Update the `download_path` variable in the notebook (Cell 1.3) to point to your data directory.

---

## Pipeline Overview

The notebook (`notebooks/perturb_paper.ipynb`) runs in 7 steps:

1. **Data Preparation** — Load ERA5, build Aurora input batch
2. **Cloud Seeding Setup** — Define seeding parameters (layers, efficiency, RH threshold)
3. **Control Forecast** — Run unperturbed Aurora rollout (14 × 6h steps = 84h), compute IVT
4. **FTLE-Guided Site Selection** — Score candidate sites using FTLE ridges, jet flanks, and AR mask
5. **Empirical Testing** — Test each candidate individually for IVT reduction at California coast
6. **Combination Optimization** — Find best multi-site seeding combinations
7. **Visualization** — IVT maps, time series, good vs. bad site comparison

---

## Custom Packages

### `aurora-seeding`
Physically consistent cloud seeding perturbations applied to Aurora model state.

```python
from aurora_seeding import apply_seeding, apply_physically_consistent_cloud_seeding
```

Key parameters: seeding layers (hPa), freeze efficiency, fallout fraction, RH threshold.

### `ftle-perturbation`
FTLE computation, candidate scoring, and Aurora rollout with perturbations.

```python
from ftle_perturbation import (
    score_perturbation_candidates,
    rollout_store_velocities_ivt,
    rollout_with_perturbation_ivt,
    DEFAULT_CONFIG,
)
```

### `ftle_calculation` (`ftle_gpu`)
GPU-accelerated FTLE field computation using RK4 trajectory integration on PyTorch.

```python
from ftle_gpu import compute_ftle

ftle_field, metadata = compute_ftle(
    u_wind=u_np,          # [T, lat, lon]
    v_wind=v_np,
    lats=lats,
    lons=lons,
    times_hours=times,
    tau_hours=72.0,
    backward=True,        # backward FTLE = attracting LCS
)
```

---

## Case Study

| Parameter | Value |
|---|---|
| AR event | 2022-12-24 12Z |
| Forecast horizon | 72–84h (steps 10–12) |
| Model | Microsoft Aurora 0.25° pretrained |
| Input | ERA5 reanalysis (two time steps: t−6h, t0) |
| IVT threshold | 500 kg/m/s |
| Best single site | 30.8°N, 166.5°E (step 0) |
| IVT reduction | 5.21% overall; ~24% of above-threshold area |
