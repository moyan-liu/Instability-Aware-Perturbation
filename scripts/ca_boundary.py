"""
California State Boundary Helper

Provides functions to load the California state boundary and perform
point-in-polygon tests for catchment calculations.
"""

import numpy as np
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.prepared import prep
from shapely.ops import unary_union

# Cache for the prepared geometry
_ca_geometry = None
_ca_prepared = None


def load_california_boundary():
    """
    Load California state boundary from Natural Earth data.

    Returns:
        shapely.geometry: California polygon geometry
    """
    global _ca_geometry, _ca_prepared

    if _ca_geometry is not None:
        return _ca_geometry

    # Load US states from Natural Earth (50m resolution)
    shpfilename = shpreader.natural_earth(
        resolution='50m',
        category='cultural',
        name='admin_1_states_provinces'
    )
    reader = shpreader.Reader(shpfilename)

    # Find California
    for record in reader.records():
        if record.attributes.get('name') == 'California':
            _ca_geometry = record.geometry
            _ca_prepared = prep(_ca_geometry)
            print(f"Loaded California boundary")
            print(f"  Bounding box: lat ({_ca_geometry.bounds[1]:.2f}, {_ca_geometry.bounds[3]:.2f}), "
                  f"lon ({_ca_geometry.bounds[0]:.2f}, {_ca_geometry.bounds[2]:.2f})")
            return _ca_geometry

    raise ValueError("California not found in Natural Earth data")


def get_ca_prepared():
    """Get the prepared (optimized) California geometry for fast queries."""
    global _ca_prepared
    if _ca_prepared is None:
        load_california_boundary()
    return _ca_prepared


def points_in_california(lons, lats):
    """
    Check which points fall within California state boundary.

    Args:
        lons: array of longitudes (can be 0-360 or -180 to 180 format)
        lats: array of latitudes

    Returns:
        boolean array of same shape as input, True if point is in CA
    """
    ca_prep = get_ca_prepared()

    # Flatten for processing
    lons_flat = np.asarray(lons).ravel()
    lats_flat = np.asarray(lats).ravel()

    # Convert 0-360 longitude to -180 to 180 if needed
    lons_converted = np.where(lons_flat > 180, lons_flat - 360, lons_flat)

    # Vectorized point-in-polygon (still need loop, but prepared geometry is fast)
    result = np.zeros(len(lons_flat), dtype=bool)

    # Quick bounding box pre-filter for efficiency
    ca_bounds = _ca_geometry.bounds  # (minx, miny, maxx, maxy)
    bbox_mask = ((lons_converted >= ca_bounds[0]) & (lons_converted <= ca_bounds[2]) &
                 (lats_flat >= ca_bounds[1]) & (lats_flat <= ca_bounds[3]))

    # Only test points within bounding box
    indices_to_test = np.where(bbox_mask)[0]

    for idx in indices_to_test:
        point = Point(lons_converted[idx], lats_flat[idx])
        result[idx] = ca_prep.contains(point)

    # Reshape to original shape
    return result.reshape(np.asarray(lons).shape)


def points_in_california_gpu(lons_gpu, lats_gpu):
    """
    GPU-compatible version: transfers to CPU, checks, transfers back.

    Args:
        lons_gpu: torch tensor of longitudes on GPU
        lats_gpu: torch tensor of latitudes on GPU

    Returns:
        torch tensor (boolean) on same device as input
    """
    import torch

    device = lons_gpu.device

    # Transfer to CPU numpy
    lons_np = lons_gpu.cpu().numpy()
    lats_np = lats_gpu.cpu().numpy()

    # Check points
    in_ca = points_in_california(lons_np, lats_np)

    # Transfer back to GPU
    return torch.tensor(in_ca, device=device, dtype=torch.bool)


def create_ca_mask_on_grid(lats, lons):
    """
    Create a 2D mask for a lat/lon grid showing which cells are in California.

    Args:
        lats: 1D array of latitudes
        lons: 1D array of longitudes (0-360 or -180 to 180 format)

    Returns:
        2D boolean array of shape (len(lats), len(lons))
    """
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return points_in_california(lon_grid, lat_grid)


# Example usage and visualization
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Load boundary
    ca_geom = load_california_boundary()

    # Create test grid
    lats = np.arange(30, 45, 0.25)
    lons = np.arange(235, 250, 0.25)  # 0-360 format

    # Create mask
    ca_mask = create_ca_mask_on_grid(lats, lons)

    # Convert lons for plotting
    lons_plot = np.where(lons > 180, lons - 360, lons)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([-130, -110, 30, 45], crs=ccrs.PlateCarree())

    # Plot mask
    lon_grid, lat_grid = np.meshgrid(lons_plot, lats)
    ax.pcolormesh(lon_grid, lat_grid, ca_mask.astype(float),
                  alpha=0.5, cmap='Blues', transform=ccrs.PlateCarree())

    # Add features
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.gridlines(draw_labels=True)

    ax.set_title('California State Boundary Mask')
    plt.savefig('ca_boundary_test.png', dpi=150, bbox_inches='tight')
    print("Saved ca_boundary_test.png")
