"""
ftle_gpu - GPU-accelerated Finite-Time Lyapunov Exponent calculation.

Provides GPU-accelerated wind interpolation, trajectory integration,
and FTLE computation using PyTorch.

Usage:
    from ftle_gpu import compute_ftle

    ftle_field, metadata = compute_ftle(
        u_wind=u_wind_np,       # [T, lat, lon] numpy array
        v_wind=v_wind_np,       # [T, lat, lon] numpy array
        lats=lats,              # [lat] array
        lons=lons,              # [lon] array
        times_hours=times_hours,# [T] array (hours since reference)
        tau_hours=72.0,
    )

Or use classes directly:
    from ftle_gpu import GPUWindInterpolator, GPUTrajectoryIntegrator, GPUFTLECalculator
"""

from .interpolator import GPUWindInterpolator
from .integrator import GPUTrajectoryIntegrator
from .calculator import GPUFTLECalculator
from .visualization import plot_ftle

__version__ = "0.1.0"

__all__ = [
    "GPUWindInterpolator",
    "GPUTrajectoryIntegrator",
    "GPUFTLECalculator",
    "plot_ftle",
    "compute_ftle",
]


def compute_ftle(
    u_wind,
    v_wind,
    lats,
    lons,
    times_hours,
    tau_hours=72.0,
    grid_spacing=0.75,
    lon_range=(0, 360),
    lat_range=(0, 70),
    dt=1.5,
    device="cuda",
    dtype=None,
    backward=False,
):
    """
    Compute the FTLE field from wind data.

    This is a convenience function that ties together GPUWindInterpolator,
    GPUTrajectoryIntegrator, and GPUFTLECalculator.

    Args:
        u_wind: [T, lat, lon] numpy array of u-wind (m/s)
        v_wind: [T, lat, lon] numpy array of v-wind (m/s)
        lats: [lat] array of latitudes (e.g. 90 to -90)
        lons: [lon] array of longitudes (e.g. 0 to 359.75)
        times_hours: [T] array of times in hours since reference
        tau_hours: Integration time in hours (default 72 = 3 days)
        grid_spacing: FTLE grid spacing in degrees (default 0.75)
        lon_range: (lon_min, lon_max) for the FTLE domain
        lat_range: (lat_min, lat_max) for the FTLE domain
        dt: RK4 time step in hours (default 1.5)
        device: "cuda" or "cpu"
        dtype: torch dtype (default torch.float64)
        backward: If True, compute backward FTLE (attracting LCS).
                  Integrates from t_start=tau_hours backward to t_end=0.
                  If False (default), compute forward FTLE (repelling LCS).

    Returns:
        ftle_field: [n_lat, n_lon] numpy array of FTLE values
        metadata: dict with grid info and timing
    """
    import time
    import numpy as np
    import torch as _torch

    if dtype is None:
        dtype = _torch.float64

    device = _torch.device(device)

    # --- Transfer wind data to device ---
    u_gpu = _torch.tensor(u_wind, device=device, dtype=dtype) if not _torch.is_tensor(u_wind) else u_wind.to(device=device, dtype=dtype)
    v_gpu = _torch.tensor(v_wind, device=device, dtype=dtype) if not _torch.is_tensor(v_wind) else v_wind.to(device=device, dtype=dtype)
    lats_gpu = _torch.tensor(lats, device=device, dtype=dtype) if not _torch.is_tensor(lats) else lats.to(device=device, dtype=dtype)
    lons_gpu = _torch.tensor(lons, device=device, dtype=dtype) if not _torch.is_tensor(lons) else lons.to(device=device, dtype=dtype)

    # --- Build particle grid ---
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range

    lons_grid = _torch.arange(lon_min, lon_max, grid_spacing, device=device, dtype=dtype)
    lats_grid = _torch.arange(lat_min, lat_max, grid_spacing, device=device, dtype=dtype)
    n_lat = int(lats_grid.numel())
    n_lon = int(lons_grid.numel())

    lon_mesh, lat_mesh = _torch.meshgrid(lons_grid, lats_grid, indexing="xy")
    initial_lon = lon_mesh.reshape(-1)
    initial_lat = lat_mesh.reshape(-1)

    # --- Create components ---
    wind_interp = GPUWindInterpolator(
        u_wind=u_gpu, v_wind=v_gpu,
        times_hours=times_hours,
        lats=lats_gpu, lons=lons_gpu,
        device=device, dtype=dtype,
    )
    integrator = GPUTrajectoryIntegrator(wind_interp, dt=dt)
    ftle_calc = GPUFTLECalculator(device, dtype=dtype)

    # --- Integrate trajectories ---
    if backward:
        t_start = tau_hours
        t_end = 0.0
    else:
        t_start = 0.0
        t_end = tau_hours

    if device.type == "cuda":
        _torch.cuda.synchronize()
    t0 = time.time()

    final_lon, final_lat = integrator.integrate(
        initial_lon, initial_lat,
        t_start=t_start, t_end=t_end,
    )

    if device.type == "cuda":
        _torch.cuda.synchronize()
    t_integration = time.time() - t0

    # --- Compute FTLE ---
    t0 = time.time()

    grad_phi = ftle_calc.compute_flow_map_gradient(
        initial_lon, initial_lat,
        final_lon, final_lat,
        grid_shape=(n_lat, n_lon),
        grid_spacing=grid_spacing,
    )
    ftle_field = ftle_calc.compute_ftle(grad_phi, lats_grid, tau_hours)

    if device.type == "cuda":
        _torch.cuda.synchronize()
    t_ftle = time.time() - t0

    # --- Return results ---
    ftle_np = ftle_field.cpu().numpy()

    metadata = {
        "lons_grid": lons_grid.cpu().numpy(),
        "lats_grid": lats_grid.cpu().numpy(),
        "n_lat": n_lat,
        "n_lon": n_lon,
        "n_particles": n_lat * n_lon,
        "tau_hours": tau_hours,
        "grid_spacing": grid_spacing,
        "dt": dt,
        "backward": backward,
        "integration_time_s": t_integration,
        "ftle_time_s": t_ftle,
        "total_time_s": t_integration + t_ftle,
    }

    return ftle_np, metadata
