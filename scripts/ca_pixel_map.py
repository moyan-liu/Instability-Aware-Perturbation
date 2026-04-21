"""
CA Pixel Map - Visualizes 0.25° grid cells selected by California mask
Shows pixel count and grid overlay for intuition.
Includes coastal strip (150 km) comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from ca_boundary import create_ca_mask_on_grid, load_california_boundary


# --- Grid setup (matching Aurora 0.25 deg resolution) ---
lats = np.arange(30.0, 44.0, 0.25)   # slightly wider than CA for context
lons = np.arange(234.0, 250.0, 0.25)  # 0-360 format (114W-126W)
lons_plot = lons - 360                 # convert to -180..180 for cartopy

ca_mask = create_ca_mask_on_grid(lats, lons)
n_pixels = int(ca_mask.sum())
print(f"Total CA pixels selected: {n_pixels}")
print(f"Grid dimensions shown: {len(lats)} lat x {len(lons)} lon")
print(f"Each pixel approx: {0.25*111:.1f} km (N-S) x {0.25*111*np.cos(np.radians(37)):.1f} km (E-W at 37N)")


# --- Coastal strip mask (pixels within strip_km of CA boundary) ---
def haversine_vec(lat1, lon1, lats2, lons2):
    """Haversine distance (km) from one point to an array of points."""
    R = 6371.0
    dlat = np.radians(lats2 - lat1)
    dlon = np.radians(lons2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lats2)) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def create_coastal_strip_mask(lats, lons_plot, ca_mask, strip_km=150):
    """Return a mask with only CA pixels within strip_km of the Pacific coastline.

    Uses the Natural Earth physical coastline (not the state political boundary)
    filtered to the CA Pacific coast region, so eastern desert pixels near the
    Nevada/Arizona border are correctly excluded.
    """
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import MultiLineString

    # Load physical coastline and filter to CA Pacific coast region
    shpfilename = shpreader.natural_earth(resolution='50m', category='physical', name='coastline')
    reader = shpreader.Reader(shpfilename)

    coast_pts = []
    for record in reader.records():
        geom = record.geometry
        # Collect all linestring coordinates
        lines = list(geom.geoms) if geom.geom_type == 'MultiLineString' else [geom]
        for line in lines:
            coords = np.array(line.coords)   # (lon, lat)
            lons_c = coords[:, 0]
            lats_c = coords[:, 1]
            # Keep only CA Pacific coast region: lat 32-42N, lon 126W-116W
            in_region = (lats_c >= 32) & (lats_c <= 42) & (lons_c >= -126) & (lons_c <= -116)
            if in_region.any():
                coast_pts.append(coords[in_region])

    coast_pts = np.vstack(coast_pts)
    b_lons = coast_pts[:, 0]
    b_lats = coast_pts[:, 1]
    print(f"  Pacific coastline points in CA region: {len(b_lons)}")

    ca_indices = np.argwhere(ca_mask)
    coastal_mask = np.zeros_like(ca_mask, dtype=bool)

    for (i, j) in ca_indices:
        lat = lats[i]
        lon = lons_plot[j]   # already -180..180
        dists = haversine_vec(lat, lon, b_lats, b_lons)
        if dists.min() <= strip_km:
            coastal_mask[i, j] = True

    return coastal_mask


STRIP_KM = 150
print(f"\nComputing {STRIP_KM} km coastal strip mask...")
coastal_mask = create_coastal_strip_mask(lats, lons_plot, ca_mask, strip_km=STRIP_KM)
n_coastal = int(coastal_mask.sum())
print(f"Coastal strip pixels ({STRIP_KM} km): {n_coastal}  ({100*n_coastal/n_pixels:.1f}% of full CA)")


# ── Helper: draw pixel rectangles ─────────────────────────────────────────
def draw_pixels(ax, mask, facecolor, edgecolor, alpha=0.5, linewidth=0.4):
    for i in range(len(lats)):
        for j in range(len(lons_plot)):
            if mask[i, j]:
                rect = mpatches.Rectangle(
                    (lons_plot[j], lats[i]), 0.25, 0.25,
                    linewidth=linewidth, edgecolor=edgecolor,
                    facecolor=facecolor, alpha=alpha,
                    transform=ccrs.PlateCarree(), zorder=2
                )
                ax.add_patch(rect)


def draw_grid(ax, extent):
    for lat in lats:
        ax.plot([extent[0], extent[1]], [lat, lat],
                color='gray', linewidth=0.25, alpha=0.4,
                transform=ccrs.PlateCarree(), zorder=1)
    for lon in lons_plot:
        ax.plot([lon, lon], [extent[2], extent[3]],
                color='gray', linewidth=0.25, alpha=0.4,
                transform=ccrs.PlateCarree(), zorder=1)


def setup_ax(ax, extent):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, color='#d0e8f5', zorder=0)
    ax.add_feature(cfeature.LAND, color='#f5f0e8', zorder=0)
    ax.add_feature(cfeature.STATES, linewidth=1.2, edgecolor='#444', zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False


# --- Figure: 3 panels ---
fig, axes = plt.subplots(1, 3, figsize=(22, 9),
                          subplot_kw={'projection': ccrs.PlateCarree()})

extent = [-126, -112, 30, 44]
for ax in axes:
    setup_ax(ax, extent)

# ── Panel 1: Full CA mask ───────────────────────────────────────────────────
ax = axes[0]
ax.set_title(f'Full CA Mask\n{n_pixels} pixels', fontsize=12, fontweight='bold')
draw_pixels(ax, ca_mask, facecolor='steelblue', edgecolor='navy')
ax.text(-119.5, 37.0, f'{n_pixels}\npixels', transform=ccrs.PlateCarree(),
        fontsize=13, fontweight='bold', color='navy', ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# ── Panel 2: Coastal strip only ─────────────────────────────────────────────
ax = axes[1]
ax.set_title(f'Coastal Strip ≤ {STRIP_KM} km\n{n_coastal} pixels  ({100*n_coastal/n_pixels:.0f}% of CA)',
             fontsize=12, fontweight='bold')
draw_pixels(ax, coastal_mask, facecolor='coral', edgecolor='darkred')
ax.text(-120.5, 37.0, f'{n_coastal}\npixels', transform=ccrs.PlateCarree(),
        fontsize=13, fontweight='bold', color='darkred', ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# ── Panel 3: Overlay comparison ─────────────────────────────────────────────
ax = axes[2]
ax.set_title(f'Comparison\nBlue = excluded inland  |  Red = coastal strip kept',
             fontsize=12, fontweight='bold')
draw_grid(ax, extent)

# Inland-only pixels (CA but NOT coastal)
inland_mask = ca_mask & ~coastal_mask
draw_pixels(ax, inland_mask, facecolor='steelblue', edgecolor='navy', alpha=0.3)
draw_pixels(ax, coastal_mask, facecolor='coral', edgecolor='darkred', alpha=0.65)

# Per-latitude-band pixel counts on the right edge
lat_bands = np.arange(32, 43, 1)
for lat_b in lat_bands:
    band_mask = (lats >= lat_b) & (lats < lat_b + 1)
    c_full = int(ca_mask[band_mask, :].sum())
    c_coast = int(coastal_mask[band_mask, :].sum())
    ax.text(-112.2, lat_b + 0.5, f'{c_coast}/{c_full}',
            transform=ccrs.PlateCarree(),
            fontsize=7, color='darkred', ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))

ax.text(-126.0, 30.5,
        f'Cell ≈ {0.25*111:.0f}×{0.25*111*np.cos(np.radians(37)):.0f} km  |  red/total per 1° band →',
        transform=ccrs.PlateCarree(), fontsize=7.5, color='black',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

plt.suptitle(f'California 0.25° Pixel Grid — Full CA vs {STRIP_KM} km Coastal Strip',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('ca_pixel_map.png', dpi=150, bbox_inches='tight')
print("\nSaved ca_pixel_map.png")
plt.show()
