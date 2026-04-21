"""
FTLE-Guided Atmospheric River Perturbation Pipeline
====================================================

# ============================================================================
# CONFIG  —  edit these paths and parameters before running
# ============================================================================

ERA5_DATA_DIR   = "/path/to/era5_batch"       # directory with daily .nc files
OUTPUT_DIR      = "results"                    # figures and summary saved here

PACKAGES_DIR    = "packages"                   # root of the packages/ folder

REFERENCE_DATE  = "2022-12-24"
REFERENCE_HOUR  = 12                           # UTC
ERA5_START_DATE = "2022-12-20"
ERA5_END_DATE   = "2023-01-15"

PIPELINE_CONFIG = {
    "event_step"            : 11,
    "landfall_window_steps" : [10, 11, 12],
    "ivt_reduction_target"  : 0.10,
    "candidates_per_step"   : 20,
    "min_separation_km"     : 500,
    "default_seeding_radius_km": 300,
}

SEEDING_PARAMS = {
    "layers_mb"           : [925, 850, 700],
    "freeze_efficiency"   : 0.30,
    "fallout_fraction"    : 0.50,
    "max_removal_fraction": 0.50,
    "min_RH"              : 0.75,
    "vertical_coupling"   : False,
    "perturb_mode"        : "both",
}

IVT_THRESHOLD   = 500.0    # kg/m/s — AR detection threshold
COASTAL_STRIP_KM = 150     # distance from Pacific coastline for CA mask

# Best site combination (rank numbers from empirical testing output)
SELECTED_RANKS  = [1, 3]

# ============================================================================
# IMPORTS
# ============================================================================

import os, sys, gc
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import MultiPolygon, Polygon

import torch

# Register local packages
for pkg in ["aurora-seeding", "ftle-perturbation", "ftle_calculation"]:
    p = os.path.join(PACKAGES_DIR, pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

from aurora import Aurora, Batch, Metadata
from aurora_seeding import apply_seeding
from aurora_seeding.physics import apply_physically_consistent_cloud_seeding, calculate_q_sat
from aurora_seeding.mask import create_seeding_mask
from ftle_perturbation import (
    DEFAULT_CONFIG,
    score_perturbation_candidates,
    compute_ca_ivt_at_event,
    compute_ca_ivt_window,
    compute_reduction,
    rollout_store_velocities_ivt,
    rollout_with_perturbation_ivt,
)
import ftle_gpu

from ca_boundary import load_california_boundary, create_ca_mask_on_grid, points_in_california

# ============================================================================
# SECTION 1 — SETUP
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Torch:  {torch.__version__}")
print(f"CUDA:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    free_gb, total_gb = [x / 1e9 for x in torch.cuda.mem_get_info()]
    print(f"GPU memory: {free_gb:.1f} / {total_gb:.1f} GB free")
    torch.cuda.empty_cache()
    gc.collect()

# ============================================================================
# SECTION 2 — LOAD ERA5 DATA
# ============================================================================

download_path = Path(ERA5_DATA_DIR)
static_file   = download_path / "static.nc"

dates       = pd.date_range(start=ERA5_START_DATE, end=ERA5_END_DATE, freq="D")
surf_files  = [download_path / f"{d.strftime('%Y-%m-%d')}-surface-level.nc" for d in dates]
atmos_files = [download_path / f"{d.strftime('%Y-%m-%d')}-atmospheric.nc" for d in dates]

static_vars_ds = xr.open_dataset(static_file, engine="netcdf4")
surf_vars_ds   = xr.open_mfdataset(surf_files,  combine="by_coords", engine="netcdf4")
atmos_vars_ds  = xr.open_mfdataset(atmos_files, combine="by_coords", engine="netcdf4")

print(f"ERA5 loaded: {ERA5_START_DATE} to {ERA5_END_DATE}")
print(f"  Surface: {dict(surf_vars_ds.sizes)}")
print(f"  Atmos:   {dict(atmos_vars_ds.sizes)}")

# ============================================================================
# SECTION 3 — IVT CALCULATION FUNCTION
# ============================================================================

def calculate_ivt_integrated(u_data, v_data, q_data, pressure_levels,
                             p_top=200, p_bottom=1000):
    """
    Vertically integrated vapor transport (IVT).

    IVT = (1/g) * integral(q * V * dp)  [kg / m / s]

    Handles 3-D (levels, lat, lon), 4-D (batch, levels, lat, lon),
    and 5-D (batch, time, levels, lat, lon) inputs.
    """
    g = 9.81
    if isinstance(pressure_levels, tuple):
        pressure_levels = torch.tensor(pressure_levels, dtype=torch.float32,
                                       device=u_data.device)

    mask          = (pressure_levels >= p_top) & (pressure_levels <= p_bottom)
    level_indices = torch.where(mask)[0]
    if len(level_indices) == 0:
        raise ValueError(f"No pressure levels between {p_top} and {p_bottom} hPa")

    pressure_levels_filtered = pressure_levels[mask]

    if len(u_data.shape) == 3:
        u_data = u_data[level_indices]
        v_data = v_data[level_indices]
        q_data = q_data[level_indices]
        n_levels, reshape_back, dp_shape, sum_dim = u_data.shape[0], None, (-1, 1, 1), 0
    elif len(u_data.shape) == 4:
        u_data = u_data[:, level_indices]
        v_data = v_data[:, level_indices]
        q_data = q_data[:, level_indices]
        n_levels, reshape_back, dp_shape, sum_dim = u_data.shape[1], None, (1, -1, 1, 1), 1
    elif len(u_data.shape) == 5:
        u_data = u_data[:, :, level_indices]
        v_data = v_data[:, :, level_indices]
        q_data = q_data[:, :, level_indices]
        n_levels     = u_data.shape[2]
        orig_shape   = u_data.shape
        u_data       = u_data.reshape(-1, *u_data.shape[2:])
        v_data       = v_data.reshape(-1, *v_data.shape[2:])
        q_data       = q_data.reshape(-1, *q_data.shape[2:])
        reshape_back = orig_shape
        dp_shape, sum_dim = (1, -1, 1, 1), 1
    else:
        raise ValueError(f"Unexpected shape: {u_data.shape}")

    pressure_pa = pressure_levels_filtered * 100
    dp = torch.zeros(n_levels, device=u_data.device, dtype=u_data.dtype)
    for i in range(n_levels - 1):
        dp[i] = abs(pressure_pa[i + 1] - pressure_pa[i])
    dp[-1] = dp[-2] if n_levels > 1 else 5000.0
    dp = dp.view(*dp_shape)

    ivt_u   = torch.sum(q_data * u_data * dp, dim=sum_dim) / g
    ivt_v   = torch.sum(q_data * v_data * dp, dim=sum_dim) / g
    ivt_mag = torch.sqrt(ivt_u**2 + ivt_v**2)

    if reshape_back is not None:
        ivt_u   = ivt_u  .reshape(reshape_back[0], reshape_back[1], *ivt_u  .shape[1:])
        ivt_v   = ivt_v  .reshape(reshape_back[0], reshape_back[1], *ivt_v  .shape[1:])
        ivt_mag = ivt_mag.reshape(reshape_back[0], reshape_back[1], *ivt_mag.shape[1:])

    return ivt_u, ivt_v, ivt_mag

# ============================================================================
# SECTION 4 — LOAD AURORA MODEL
# ============================================================================

print("Loading Aurora model...")
model = Aurora(use_lora=False)
model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
model.eval()
model = model.to("cuda")
print("Aurora model loaded")

# ============================================================================
# SECTION 5 — BUILD AURORA INPUT BATCH
# ============================================================================

ref_datetime    = pd.Timestamp(f"{REFERENCE_DATE} {REFERENCE_HOUR:02d}:00:00")
available_times = pd.DatetimeIndex(surf_vars_ds.valid_time.values)
time_index_current  = next(i for i, t in enumerate(available_times) if t == ref_datetime)
time_index_previous = next(i for i, t in enumerate(available_times)
                           if t == ref_datetime - timedelta(hours=6))

print(f"Reference: {ref_datetime}")
print(f"ERA5 input times: {available_times[time_index_previous]} → {available_times[time_index_current]}")

lat_crop   = 720
lon_crop   = 1440
time_slice = slice(time_index_previous, time_index_current + 1)

batch_control = Batch(
    surf_vars={
        "2t" : torch.from_numpy(surf_vars_ds["t2m"].values[time_slice, :lat_crop, :lon_crop][None]),
        "10u": torch.from_numpy(surf_vars_ds["u10"].values[time_slice, :lat_crop, :lon_crop][None]),
        "10v": torch.from_numpy(surf_vars_ds["v10"].values[time_slice, :lat_crop, :lon_crop][None]),
        "msl": torch.from_numpy(surf_vars_ds["msl"].values[time_slice, :lat_crop, :lon_crop][None]),
    },
    static_vars={
        "z"  : torch.from_numpy(static_vars_ds["z"]  .values[0, :lat_crop, :lon_crop]),
        "slt": torch.from_numpy(static_vars_ds["slt"].values[0, :lat_crop, :lon_crop]),
        "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0, :lat_crop, :lon_crop]),
    },
    atmos_vars={
        "t": torch.from_numpy(atmos_vars_ds["t"].values[time_slice, :, :lat_crop, :lon_crop][None]),
        "u": torch.from_numpy(atmos_vars_ds["u"].values[time_slice, :, :lat_crop, :lon_crop][None]),
        "v": torch.from_numpy(atmos_vars_ds["v"].values[time_slice, :, :lat_crop, :lon_crop][None]),
        "q": torch.from_numpy(atmos_vars_ds["q"].values[time_slice, :, :lat_crop, :lon_crop][None]),
        "z": torch.from_numpy(atmos_vars_ds["z"].values[time_slice, :, :lat_crop, :lon_crop][None]),
    },
    metadata=Metadata(
        lat=torch.from_numpy(surf_vars_ds.latitude .values[:lat_crop]),
        lon=torch.from_numpy(surf_vars_ds.longitude.values[:lon_crop]),
        time=(available_times[time_index_current].to_pydatetime(),),
        atmos_levels=tuple(int(l) for l in atmos_vars_ds.pressure_level.values),
    ),
).to("cuda")

print(f"Batch created: {batch_control.spatial_shape}")
print(f"Atmos levels:  {batch_control.metadata.atmos_levels}")

# ============================================================================
# SECTION 6 — PIPELINE CONFIGURATION
# ============================================================================

cfg = {**DEFAULT_CONFIG, **PIPELINE_CONFIG}

pressure_levels      = batch_control.metadata.atmos_levels
pressure_levels_list = list(pressure_levels)
wind_level_idx       = pressure_levels_list.index(cfg["wind_level"])
jet_level_idx        = pressure_levels_list.index(cfg["jet_level"])
num_steps            = max(14, cfg["landfall_window_steps"][-1] + 2)

print(f"\nPipeline config:")
print(f"  FTLE level: {cfg['wind_level']} hPa  |  Jet level: {cfg['jet_level']} hPa")
print(f"  IVT threshold: {cfg['ivt_threshold']} kg/(m·s)")
print(f"  Forecast steps: {num_steps} ({num_steps * 6} hr)")
print(f"  Landfall window steps: {cfg['landfall_window_steps']}")

# ============================================================================
# SECTION 7 — COASTAL STRIP MASK
# ============================================================================

lats_np = batch_control.metadata.lat.cpu().numpy()
lons_np = batch_control.metadata.lon.cpu().numpy()


def _build_coastal_strip_mask(lats, lons, strip_km=COASTAL_STRIP_KM):
    """Boolean mask: Aurora grid pixels within strip_km of the CA Pacific coast."""
    def _haversine(lat1, lon1, lats2, lons2):
        R    = 6371.0
        dlat = np.radians(lats2 - lat1)
        dlon = np.radians(lons2 - lon1)
        a    = (np.sin(dlat / 2)**2
                + np.cos(np.radians(lat1)) * np.cos(np.radians(lats2)) * np.sin(dlon / 2)**2)
        return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    shpfile   = shpreader.natural_earth(resolution="50m", category="physical", name="coastline")
    coast_pts = []
    for rec in shpreader.Reader(shpfile).records():
        geom  = rec.geometry
        lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for line in lines:
            coords = np.array(line.coords)
            lc, lt = coords[:, 0], coords[:, 1]
            sel = (lt >= 32) & (lt <= 42) & (lc >= -126) & (lc <= -116)
            if sel.any():
                coast_pts.append(coords[sel])

    coast_pts = np.vstack(coast_pts)
    b_lons, b_lats = coast_pts[:, 0], coast_pts[:, 1]

    ca_full  = create_ca_mask_on_grid(lats, lons)
    lons_180 = np.where(lons > 180, lons - 360, lons)
    coastal  = np.zeros_like(ca_full, dtype=bool)
    for (i, j) in np.argwhere(ca_full):
        if _haversine(lats[i], lons_180[j], b_lats, b_lons).min() <= strip_km:
            coastal[i, j] = True
    return coastal


print(f"Building {COASTAL_STRIP_KM} km coastal strip mask...")
_coastal_strip_cache = _build_coastal_strip_mask(lats_np, lons_np)
print(f"  {_coastal_strip_cache.sum()} pixels in coastal strip")


def coastal_mask_fn(lats, lons):
    return _coastal_strip_cache

# ============================================================================
# SECTION 8 — CONTROL FORECAST
# ============================================================================

print("\nRunning control forecast...")
with torch.no_grad():
    preds_ctrl, ivt_ctrl, vel_fields_ctrl = rollout_store_velocities_ivt(
        model, batch_control, steps=num_steps,
        pressure_levels=pressure_levels,
        ivt_fn=calculate_ivt_integrated,
    )

model.to("cpu")
torch.cuda.empty_cache()

ctrl_ca_ivt = compute_ca_ivt_window(
    ivt_ctrl, cfg["landfall_window_steps"], lats_np, lons_np, coastal_mask_fn
)
print(f"Control forecast: {len(preds_ctrl)} steps")
print(f"Control CA IVT (landfall window): {ctrl_ca_ivt:.1f} kg/(m·s)")

# ============================================================================
# SECTION 9 — FTLE-GUIDED CANDIDATE SCORING
# ============================================================================

print("\nScoring FTLE-guided perturbation candidates...")
all_candidates, all_ftle, all_catchment, all_meta = score_perturbation_candidates(
    ivt_ctrl, vel_fields_ctrl, lats_np, lons_np,
    ca_boundary_fn=points_in_california,
    wind_level_idx=wind_level_idx,
    jet_level_idx=jet_level_idx,
    ftle_gpu=ftle_gpu,
    config=cfg,
)

if not all_candidates:
    raise RuntimeError("No viable perturbation candidates found.")

all_candidates = [c for c in all_candidates if c["step"] < cfg["event_step"] - 10]
print(f"Candidates after propagation-time filter: {len(all_candidates)}")

# ============================================================================
# SECTION 10 — INDIVIDUAL CANDIDATE TESTING
# ============================================================================

print("\n" + "=" * 70)
print("EMPIRICAL CANDIDATE TESTING — IVT reduction over landfall window")
print("=" * 70)

candidate_results = []

for i, cand in enumerate(all_candidates):
    print(f"[{i+1:2d}/{len(all_candidates)}] "
          f"Step {cand['step']:2d} (+{(cand['step']+1)*6}hr) "
          f"({cand['lat']:5.1f}N, {cand['lon']:6.1f}E) "
          f"FTLE={cand['ftle_value']:.3f} ... ", end="", flush=True)

    single_schedule = [{"step": cand["step"], "lat": cand["lat"],
                        "lon": cand["lon"], "radius_km": cfg["default_seeding_radius_km"]}]

    model.to("cuda")
    batch_test = Batch(
        surf_vars  ={k: v.clone() for k, v in batch_control.surf_vars  .items()},
        atmos_vars ={k: v.clone() for k, v in batch_control.atmos_vars .items()},
        static_vars={k: v.clone() for k, v in batch_control.static_vars.items()},
        metadata=batch_control.metadata,
    ).to("cuda")

    with torch.no_grad():
        _, ivt_test, _ = rollout_with_perturbation_ivt(
            model, batch_test, num_steps,
            single_schedule, SEEDING_PARAMS,
            create_seeding_mask_fn=create_seeding_mask,
            apply_seeding_fn=apply_physically_consistent_cloud_seeding,
            pressure_levels=pressure_levels,
            ivt_fn=calculate_ivt_integrated,
        )

    model.to("cpu")
    torch.cuda.empty_cache()

    test_ca_ivt = compute_ca_ivt_window(
        ivt_test, cfg["landfall_window_steps"], lats_np, lons_np, coastal_mask_fn
    )
    reduction = compute_reduction(ctrl_ca_ivt, test_ca_ivt)

    candidate_results.append({
        "candidate_idx": i,
        "step"         : cand["step"],
        "hour"         : (cand["step"] + 1) * 6,
        "lat"          : cand["lat"],
        "lon"          : cand["lon"],
        "ftle_value"   : cand["ftle_value"],
        "ivt_value"    : cand["ivt_value"],
        "jet_speed"    : cand["jet_speed"],
        "proxy_score"  : cand["score"],
        "test_ca_ivt"  : test_ca_ivt,
        "reduction"    : reduction,
        "reduction_pct": reduction * 100,
    })

    status = "REDUCES" if reduction > 0.01 else ("increases" if reduction < -0.01 else "neutral")
    print(f"IVT={test_ca_ivt:.1f}  delta={reduction*100:+.2f}% {status}")

# ============================================================================
# SECTION 11 — EMPIRICAL RANKING
# ============================================================================

ranked_results   = sorted(candidate_results, key=lambda x: x["reduction"], reverse=True)
top_candidates   = [r for r in ranked_results if r["reduction"] > 0][:10]

print(f"\n{'Rank':<5} {'Step':<5} {'Hour':<6} {'Lat':>7} {'Lon':>8} "
      f"{'FTLE':>7} {'Jet':>5} {'Proxy':>7} {'IVT Reduc':>10} {'Status'}")
print("-" * 85)
for rank, r in enumerate(ranked_results, 1):
    status = "GOOD" if r["reduction"] > 0.005 else ("BAD" if r["reduction"] < -0.005 else "neutral")
    print(f"{rank:<5} {r['step']:<5} +{r['hour']:<5} {r['lat']:>7.1f} {r['lon']:>8.1f} "
          f"{r['ftle_value']:>7.4f} {r['jet_speed']:>5.0f} {r['proxy_score']:>7.4f} "
          f"{r['reduction_pct']:>+9.2f}% {status}")

# ============================================================================
# SECTION 12 — COMBINATION TEST (user-selected ranks)
# ============================================================================

print(f"\nTesting combination: ranks {SELECTED_RANKS}")
selected       = [ranked_results[r - 1] for r in SELECTED_RANKS]
combo_schedule = [{"step": r["step"], "lat": r["lat"], "lon": r["lon"],
                   "radius_km": cfg["default_seeding_radius_km"]} for r in selected]
sum_individual = sum(r["reduction_pct"] for r in selected)

model.to("cuda")
batch_combo = Batch(
    surf_vars  ={k: v.clone() for k, v in batch_control.surf_vars  .items()},
    atmos_vars ={k: v.clone() for k, v in batch_control.atmos_vars .items()},
    static_vars={k: v.clone() for k, v in batch_control.static_vars.items()},
    metadata=batch_control.metadata,
).to("cuda")

with torch.no_grad():
    preds_best, ivt_best, _ = rollout_with_perturbation_ivt(
        model, batch_combo, num_steps,
        combo_schedule, SEEDING_PARAMS,
        create_seeding_mask_fn=create_seeding_mask,
        apply_seeding_fn=apply_physically_consistent_cloud_seeding,
        pressure_levels=pressure_levels,
        ivt_fn=calculate_ivt_integrated,
    )

model.to("cpu")
torch.cuda.empty_cache()

best_ca_ivt      = compute_ca_ivt_window(
    ivt_best, cfg["landfall_window_steps"], lats_np, lons_np, coastal_mask_fn
)
best_reduction   = compute_reduction(ctrl_ca_ivt, best_ca_ivt) * 100
synergy          = best_reduction / sum_individual if sum_individual != 0 else float("nan")

print(f"  Control IVT  : {ctrl_ca_ivt:.1f} kg/(m·s)")
print(f"  Perturbed IVT: {best_ca_ivt:.1f} kg/(m·s)")
print(f"  Reduction    : {best_reduction:+.2f}%")
print(f"  Synergy      : {synergy:.2f}x")

# ============================================================================
# SECTION 13 — ERA5 REFERENCE IVT TIME SERIES
# ============================================================================

n_steps        = len(ivt_ctrl)
forecast_times = [ref_datetime + pd.Timedelta(hours=(i + 1) * 6) for i in range(n_steps)]

era5_lats_raw  = atmos_vars_ds.coords["latitude"] .values
era5_lons_raw  = atmos_vars_ds.coords["longitude"].values
era5_levels    = atmos_vars_ds.coords["pressure_level"].values
flip_lat       = era5_lats_raw[0] > era5_lats_raw[-1]
era5_lats_asc  = era5_lats_raw[::-1] if flip_lat else era5_lats_raw

ivt_era5 = []
for i, t in enumerate(forecast_times):
    try:
        ds_t = atmos_vars_ds.sel(valid_time=t, method="nearest")
        u_t  = torch.tensor(ds_t["u"].values, dtype=torch.float32)
        v_t  = torch.tensor(ds_t["v"].values, dtype=torch.float32)
        q_t  = torch.tensor(ds_t["q"].values, dtype=torch.float32)
        pl   = torch.tensor(era5_levels.tolist(), dtype=torch.float32)
        _, _, ivt_mag = calculate_ivt_integrated(u_t, v_t, q_t, pl)
        ivt_np = ivt_mag.numpy()
        if flip_lat:
            ivt_np = ivt_np[::-1, :]
        interp = RegularGridInterpolator(
            (era5_lats_asc, era5_lons_raw), ivt_np,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        lon2d, lat2d = np.meshgrid(lons_np, lats_np)
        ivt_era5.append(interp((lat2d, lon2d)))
    except Exception as e:
        print(f"  ERA5 step {i} ({t}): failed — {e}")
        ivt_era5.append(np.full_like(ivt_ctrl[0], np.nan))

# ============================================================================
# SECTION 14 — AREA-WEIGHTED TIME SERIES
# ============================================================================

ca_mask    = _coastal_strip_cache
lat2d      = np.broadcast_to(np.cos(np.radians(lats_np))[:, None], ca_mask.shape)
ca_weights = np.where(ca_mask, lat2d, 0.0)
w_sum      = ca_weights.sum()
hours_arr  = np.arange(1, n_steps + 1) * 6

ca_ivt_ctrl_ts = np.array([np.sum(ivt_ctrl[i]  * ca_weights) / w_sum for i in range(n_steps)])
ca_ivt_pert_ts = np.array([np.sum(ivt_best[i]  * ca_weights) / w_sum for i in range(n_steps)])
ca_ivt_era5_ts = np.array([np.sum(ivt_era5[i]  * ca_weights) / w_sum for i in range(n_steps)])
ca_ivt_anom_ts = ca_ivt_pert_ts - ca_ivt_ctrl_ts
ca_ivt_pct_ts  = np.where(ca_ivt_ctrl_ts > 0, ca_ivt_anom_ts / ca_ivt_ctrl_ts * 100, np.nan)

valid_steps  = [s for s in cfg["landfall_window_steps"] if s < n_steps]
lf_hours     = [(s + 1) * 6 for s in valid_steps]

# Above-threshold exceedance reduction (Option B)
ivt_mean_ctrl = np.mean([ivt_ctrl[s] for s in valid_steps], axis=0)
ivt_mean_pert = np.mean([ivt_best[s] for s in valid_steps], axis=0)
ctrl_excess   = np.sum(np.maximum(ivt_mean_ctrl[ca_mask] - IVT_THRESHOLD, 0))
pert_excess   = np.sum(np.maximum(ivt_mean_pert[ca_mask] - IVT_THRESHOLD, 0))
scalar_b      = (ctrl_excess - pert_excess) / ctrl_excess * 100 if ctrl_excess > 0 else np.nan

print(f"\nAbove-threshold IVT reduction (Option B): {scalar_b:+.2f}%")

# ============================================================================
# SECTION 15 — FIGURES
# ============================================================================

ca_lons_180        = np.where(lons_np > 180, lons_np - 360, lons_np)
lon2d_plot, lat2d_plot = np.meshgrid(ca_lons_180, lats_np)

mask_lon_idx = np.where(ca_mask.any(axis=0))[0]
mask_lat_idx = np.where(ca_mask.any(axis=1))[0]
ca_extent    = [
    ca_lons_180[mask_lon_idx].min() - 1, ca_lons_180[mask_lon_idx].max() + 1,
    lats_np    [mask_lat_idx].min() - 1, lats_np    [mask_lat_idx].max() + 1,
]

# --- Figure 1: Candidate performance bar chart + site map -------------------

sorted_cands = sorted(candidate_results, key=lambda x: x["reduction"], reverse=True)
reductions   = [r["reduction_pct"] for r in sorted_cands]
bar_colors   = ["green" if r > 0 else "red" for r in reductions]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
ax1.bar(range(len(reductions)), reductions, color=bar_colors, alpha=0.7,
        edgecolor="black", linewidth=0.5)
ax1.axhline(0, color="black", linewidth=1)
ax1.axhline(cfg["ivt_reduction_target"] * 100, color="blue", linestyle="--", linewidth=2,
            label=f"Target ({cfg['ivt_reduction_target']*100:.0f}%)")
ax1.set_xlabel("Candidate (ranked by reduction)", fontsize=11)
ax1.set_ylabel("IVT Reduction (%)", fontsize=11)
ax1.set_title("Individual Candidate Performance\n(green = reduces IVT)", fontsize=12, fontweight="bold")
ax1.legend()
ax1.set_xlim(-1, len(sorted_cands))

ax_map = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree(central_longitude=180))
ax_map.add_feature(cfeature.LAND,      facecolor="lightgray", alpha=0.5)
ax_map.add_feature(cfeature.OCEAN,     facecolor="lightblue", alpha=0.3)
ax_map.add_feature(cfeature.COASTLINE, linewidth=1.0)
ax_map.set_extent([150, 260, 10, 70], crs=ccrs.PlateCarree())

ca_geom = load_california_boundary()
for poly in (ca_geom.geoms if isinstance(ca_geom, MultiPolygon) else [ca_geom]):
    x, y = poly.exterior.xy
    ax_map.plot(x, y, "r-", linewidth=2, transform=ccrs.PlateCarree())
    ax_map.fill(x, y, color="red", alpha=0.15, transform=ccrs.PlateCarree())

if top_candidates:
    max_red = max(r["reduction_pct"] for r in top_candidates)
    for i, r in enumerate(top_candidates, 1):
        intensity = r["reduction_pct"] / max_red if max_red > 0 else 0
        color     = plt.cm.Greens(0.3 + 0.7 * intensity)
        ax_map.scatter(r["lon"], r["lat"], c=[color], s=150, marker="o",
                       edgecolor="black", linewidth=1.5,
                       transform=ccrs.PlateCarree(), zorder=10)
        ax_map.annotate(str(i), (r["lon"], r["lat"]), xytext=(5, 5),
                        textcoords="offset points", fontsize=10, fontweight="bold",
                        transform=ccrs.PlateCarree())

ax_map.set_title("Empirically-Ranked Candidates\n(darker green = higher reduction)",
                 fontsize=12, fontweight="bold")
gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle="--")
gl.top_labels = gl.right_labels = False

plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, f"candidates_{ref_datetime.strftime('%Y%m%d_%H')}.png")
plt.savefig(fig1_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {fig1_path}")
plt.close()

# --- Figure 2: IVT time series + per-pixel change maps at landfall ----------

step_maps = {}
for s in valid_steps:
    ar_mask_s = ca_mask & (ivt_ctrl[s] > IVT_THRESHOLD)
    step_maps[s] = np.where(
        ar_mask_s,
        np.where(ivt_ctrl[s] > 0,
                 (ivt_ctrl[s] - ivt_best[s]) / ivt_ctrl[s] * 100,
                 np.nan),
        np.nan,
    )

all_vals = np.concatenate([step_maps[s][~np.isnan(step_maps[s])] for s in valid_steps])
vlim     = np.percentile(np.abs(all_vals), 97) if len(all_vals) > 0 else 5.0

n_maps = len(valid_steps)
fig    = plt.figure(figsize=(16, 11))
gs     = gridspec.GridSpec(2, n_maps, figure=fig, height_ratios=[1.1, 1.3],
                            hspace=0.40, wspace=0.35)

ax_ts = fig.add_subplot(gs[0, :])
ax_ts.axvspan(min(lf_hours) - 3, max(lf_hours) + 3,
              color="gold", alpha=0.25, zorder=0, label="Landfall window")
ax_ts.plot(hours_arr, ca_ivt_era5_ts, "g^--", lw=2, ms=7, label="ERA5 (observed)")
ax_ts.plot(hours_arr, ca_ivt_ctrl_ts, "b^--", lw=2, ms=7, label="Aurora control")
ax_ts.plot(hours_arr, ca_ivt_pert_ts, "rs-",  lw=2, ms=7, label="Aurora perturbed")
ax_ts.fill_between(hours_arr, ca_ivt_ctrl_ts, ca_ivt_pert_ts,
                   where=(ca_ivt_anom_ts >= 0), color="red",  alpha=0.15)
ax_ts.fill_between(hours_arr, ca_ivt_ctrl_ts, ca_ivt_pert_ts,
                   where=(ca_ivt_anom_ts <  0), color="blue", alpha=0.15)
ax_ts.set_xlabel("Forecast Hour", fontsize=12, fontweight="bold")
ax_ts.set_ylabel("Area-Mean IVT [kg/(m·s)]", fontsize=12, fontweight="bold")
ax_ts.set_title(
    f"CA Coastal Strip Area-Mean IVT ({COASTAL_STRIP_KM} km from Pacific Coastline)\n"
    f"Aurora Init: {ref_datetime.strftime('%Y-%m-%d %HZ')}",
    fontsize=13, fontweight="bold",
)
ax_ts.legend(fontsize=11)
ax_ts.grid(True, alpha=0.3, ls="--")

for col, s in enumerate(valid_steps):
    ax_map = fig.add_subplot(gs[1, col], projection=ccrs.PlateCarree())
    ax_map.set_extent(ca_extent, crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.OCEAN,     color="#d0e8f5",    zorder=0)
    ax_map.add_feature(cfeature.LAND,      facecolor="#f5f0e8", alpha=0.6, zorder=0)
    ax_map.add_feature(cfeature.COASTLINE, lw=1,               zorder=3)
    ax_map.add_feature(cfeature.STATES,    lw=0.8, linestyle=":", zorder=3)
    gl = ax_map.gridlines(draw_labels=True, lw=0.3, alpha=0.4, linestyle="--")
    gl.top_labels = gl.right_labels = False

    cf = ax_map.pcolormesh(lon2d_plot, lat2d_plot, step_maps[s],
                           cmap="RdBu", vmin=-vlim, vmax=vlim,
                           transform=ccrs.PlateCarree(), shading="auto", zorder=2)
    n_ar      = int((ca_mask & (ivt_ctrl[s] > IVT_THRESHOLD)).sum())
    step_mean = np.nanmean(step_maps[s])
    ax_map.set_title(f"Step {s}  (+{(s+1)*6}hr)\n"
                     f"Mean: {step_mean:+.2f}%  |  {n_ar} AR pixels",
                     fontsize=11, fontweight="bold")
    plt.colorbar(cf, ax=ax_map, label="IVT Change [%]\n(+ = reduced, blue)",
                 shrink=0.75, pad=0.04, extend="both")

fig.suptitle(
    f"Per-Pixel IVT Change at Each Landfall Step  |  "
    f"Aurora Init {ref_datetime.strftime('%Y-%m-%d %HZ')}\n"
    f"Coastal strip, AR pixels only (ctrl > {IVT_THRESHOLD:.0f} kg/m·s)  |  "
    f"Above-threshold IVT reduction: {scalar_b:+.2f}%",
    fontsize=13, fontweight="bold", y=1.01,
)

fig2_path = os.path.join(OUTPUT_DIR,
    f"ivt_analysis_{ref_datetime.strftime('%Y%m%d_%H')}_thr{int(IVT_THRESHOLD)}.png")
plt.savefig(fig2_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {fig2_path}")
plt.close()

# ============================================================================
# SECTION 16 — SUMMARY TABLE
# ============================================================================

print(f"\n{'Hr':>5}  {'ERA5':>10}  {'Control':>10}  {'Perturbed':>10}  {'Δ IVT':>8}  {'Δ%':>7}")
print("-" * 60)
for i in range(n_steps):
    marker = " <-- LANDFALL" if i in valid_steps else ""
    print(f"+{hours_arr[i]:<4}  {ca_ivt_era5_ts[i]:>10.2f}  {ca_ivt_ctrl_ts[i]:>10.2f}  "
          f"{ca_ivt_pert_ts[i]:>10.2f}  {ca_ivt_anom_ts[i]:>+8.2f}  "
          f"{ca_ivt_pct_ts[i]:>+6.1f}%{marker}")
print("-" * 60)

lf_ctrl = np.mean([ca_ivt_ctrl_ts[s] for s in valid_steps])
lf_pert = np.mean([ca_ivt_pert_ts[s] for s in valid_steps])
lf_anom = lf_pert - lf_ctrl
print(f"{'LF mean':>7}  {'':>10}  {lf_ctrl:>10.2f}  {lf_pert:>10.2f}  "
      f"{lf_anom:>+8.2f}  {lf_anom/lf_ctrl*100:>+6.1f}%  (area-mean, coastal strip)")
print(f"{'LF Opt-B':>7}  {'':>10}  {'':>10}  {'':>10}  {'':>8}  "
      f"{scalar_b:>+6.1f}%  (above-threshold exceedance, {IVT_THRESHOLD:.0f} kg/m·s)")

print(f"\nDone. Results saved to {OUTPUT_DIR}/")
