# FTLE-Guided Atmospheric River Perturbation

Research code for identifying optimal cloud seeding locations to reduce landfalling Atmospheric River (AR) precipitation using the [Microsoft Aurora](https://github.com/microsoft/aurora) weather model and Finite-Time Lyapunov Exponents (FTLE).
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

## Data

ERA5 reanalysis data is required but **not included** in this repository.


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
## Citation

If you use this repository or build on this workflow, please cite:

**Instability-Aware Steering of an Extreme Atmospheric River in an AI Weather Foundation Model**  
Moyan Liu, Qin Huang, Upmanu Lall  
arXiv:2604.18906 [physics.ao-ph]
