"""FTLE field visualization with cartopy map projections."""

import numpy as np


def plot_ftle(ftle_field, lons_grid, lats_grid, title=None,
              cmap="Blues", percentile_clip=(5, 95), figsize=(14, 8),
              central_longitude=180, save_path=None):
    """
    Plot an FTLE field on a map projection.

    Args:
        ftle_field: [n_lat, n_lon] numpy array of FTLE values
        lons_grid: [n_lon] array of longitudes
        lats_grid: [n_lat] array of latitudes
        title: Plot title (optional)
        cmap: Matplotlib colormap name
        percentile_clip: (low, high) percentiles for color scaling
        figsize: Figure size tuple
        central_longitude: Central longitude for PlateCarree projection
        save_path: If provided, save figure to this path

    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Convert torch tensors to numpy if needed
    if hasattr(ftle_field, "cpu"):
        ftle_field = ftle_field.cpu().numpy()
    if hasattr(lons_grid, "cpu"):
        lons_grid = lons_grid.cpu().numpy()
    if hasattr(lats_grid, "cpu"):
        lats_grid = lats_grid.cpu().numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=central_longitude))

    lon_min, lon_max = float(lons_grid[0]), float(lons_grid[-1])
    lat_min, lat_max = float(lats_grid[0]), float(lats_grid[-1])
    # Ensure lat_min < lat_max
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--")
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    lon_mesh, lat_mesh = np.meshgrid(lons_grid, lats_grid)

    vmin = np.percentile(ftle_field, percentile_clip[0])
    vmax = np.percentile(ftle_field, percentile_clip[1])

    im = ax.pcolormesh(
        lon_mesh, lat_mesh, ftle_field,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        shading="auto",
    )

    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
    cbar.set_label("FTLE (1/day)", fontsize=11)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax
