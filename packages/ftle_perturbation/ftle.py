"""FTLE and catchment map computation using GPU-accelerated trajectory integration."""

import numpy as np
import torch


def compute_ftle_and_catchment(vel_fields, lats, lons,
                               ca_boundary_fn,
                               t_start_step, wind_level_idx,
                               grid_spacing=0.75,
                               lon_range=(150, 260), lat_range=(10, 60),
                               ftle_gpu=None):
    """Compute forward FTLE and catchment map from a single trajectory integration.

    Uses ftle_gpu classes to integrate particles forward and compute both
    the FTLE field and catchment labels from the same final positions.

    Args:
        vel_fields: list of dicts {'u', 'v', 'q'} as numpy [levels, lat, lon]
        lats: 1-D numpy array of latitudes
        lons: 1-D numpy array of longitudes
        ca_boundary_fn: callable(lon_array, lat_array) -> boolean mask
            Determines which final positions land in the California target.
            Typically ``points_in_california`` from ca_boundary.
        t_start_step: forecast step to begin integration from
        wind_level_idx: index into the level dimension for u/v extraction
        grid_spacing: FTLE grid spacing in degrees
        lon_range: (lon_min, lon_max) for the particle grid
        lat_range: (lat_min, lat_max) for the particle grid
        ftle_gpu: the ``ftle_gpu`` module (must provide GPUWindInterpolator,
            GPUTrajectoryIntegrator, GPUFTLECalculator)

    Returns:
        ftle_field: 2-D numpy array (n_lat, n_lon) or None
        catchment_mask: 2-D boolean numpy array (n_lat, n_lon) or None
        metadata: dict with grid info and final positions, or None
    """
    if ftle_gpu is None:
        raise ValueError("ftle_gpu module must be provided")

    n_steps = len(vel_fields) - t_start_step
    if n_steps < 2:
        return None, None, None

    # Extract wind at specified level
    u_wind_list = []
    v_wind_list = []
    for i in range(t_start_step, len(vel_fields)):
        u_wind_list.append(vel_fields[i]['u'][wind_level_idx])
        v_wind_list.append(vel_fields[i]['v'][wind_level_idx])

    u_wind = np.stack(u_wind_list, axis=0)
    v_wind = np.stack(v_wind_list, axis=0)
    times_hours = np.arange(n_steps) * 6.0
    tau_hours = times_hours[-1]

    if tau_hours < 6:
        return None, None, None

    # Convert to GPU tensors
    u_gpu = torch.tensor(u_wind, device="cuda", dtype=torch.float64)
    v_gpu = torch.tensor(v_wind, device="cuda", dtype=torch.float64)
    lats_gpu = torch.tensor(lats, device="cuda", dtype=torch.float64)
    lons_gpu = torch.tensor(lons, device="cuda", dtype=torch.float64)

    # Create interpolator
    interp = ftle_gpu.GPUWindInterpolator(
        u_gpu, v_gpu, times_hours, lats_gpu, lons_gpu,
        device="cuda", dtype=torch.float64
    )

    # Create particle grid
    grid_lons = np.arange(lon_range[0], lon_range[1], grid_spacing)
    grid_lats = np.arange(lat_range[0], lat_range[1], grid_spacing)
    n_lon = len(grid_lons)
    n_lat = len(grid_lats)

    lon_grid_2d, lat_grid_2d = np.meshgrid(grid_lons, grid_lats)
    initial_lon = torch.tensor(lon_grid_2d.ravel(), dtype=torch.float64, device="cuda")
    initial_lat = torch.tensor(lat_grid_2d.ravel(), dtype=torch.float64, device="cuda")

    # Single forward integration pass
    integrator = ftle_gpu.GPUTrajectoryIntegrator(interp, dt=1.5)
    final_lon, final_lat = integrator.integrate(initial_lon, initial_lat, 0.0, tau_hours)

    # Compute FTLE from flow map gradient
    calculator = ftle_gpu.GPUFTLECalculator(device="cuda", dtype=torch.float64)
    grad_phi = calculator.compute_flow_map_gradient(
        initial_lon, initial_lat, final_lon, final_lat,
        grid_shape=(n_lat, n_lon), grid_spacing=grid_spacing
    )
    lats_grid_tensor = torch.tensor(grid_lats, dtype=torch.float64, device="cuda")
    ftle_field = calculator.compute_ftle(grad_phi, lats_grid_tensor, tau_hours)
    ftle_field = ftle_field.cpu().numpy()

    # Compute catchment from same final positions
    final_lon_np = final_lon.cpu().numpy()
    final_lat_np = final_lat.cpu().numpy()

    in_ca = ca_boundary_fn(final_lon_np, final_lat_np)
    catchment_mask = in_ca.reshape(n_lat, n_lon)

    metadata = {
        'grid_lons': grid_lons,
        'grid_lats': grid_lats,
        'n_lat': n_lat,
        'n_lon': n_lon,
        'tau_hours': tau_hours,
        't_start_step': t_start_step,
        'final_lon': final_lon_np.reshape(n_lat, n_lon),
        'final_lat': final_lat_np.reshape(n_lat, n_lon),
    }

    return ftle_field, catchment_mask, metadata
