print("📊 Creating single timestep IVT anomaly map...")

# ========== SELECT WHICH TIMESTEP TO PLOT ==========
PLOT_TIMESTEP = 0  # Change this: 0 = +6hr, 1 = +12hr, 2 = +18hr, etc.
# Or specify hours directly:
# PLOT_HOUR = 24  # Will find closest timestep to +24hr

# ---------- Setup ----------
proj = ccrs.PlateCarree(central_longitude=180)
data_crs = ccrs.PlateCarree()
extent = [60, 300, 0, 90]  # NH focus

lon_grid, lat_grid = np.meshgrid(lons_np, lats_np)

# NaN-safe copies
ivt_diff = [np.array(a, dtype=float) for a in ivt_diff]
n_steps = len(ivt_diff)

# ---------- Color scale ----------
amax = 400  # Fixed scale
levels = np.linspace(-amax, amax, 21)

# ---------- Helper: geodesic circle ----------
def plot_geodesic_circle(ax, lon, lat, radius_km, n=180, **kwargs):
    """Plot a geodesic circle (filled or outline)"""
    from cartopy.geodesic import Geodesic
    from matplotlib.patches import Polygon
    g = Geodesic()
    poly = g.circle(lon=lon, lat=lat, radius=radius_km*1000.0, n_samples=n)
    xs, ys = zip(*poly)
    polygon = Polygon(list(zip(xs, ys)), transform=ccrs.Geodetic(), **kwargs)
    ax.add_patch(polygon)
    return polygon

# ---------- SELECT TIMESTEP ----------
idx = PLOT_TIMESTEP
if idx >= n_steps:
    print(f"⚠️  Warning: Timestep {idx} exceeds available data (max: {n_steps-1})")
    idx = n_steps - 1

hours = (idx+1)*6
print(f"  Plotting timestep {idx} (+{hours} hours)")

# ========== CREATE SINGLE FIGURE ==========
fig, ax = plt.subplots(
    figsize=(16, 10),
    subplot_kw={'projection': proj}
)

# Base map
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.4, zorder=1)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.25, zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black', zorder=4)
ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='gray', alpha=0.5, zorder=4)
ax.set_extent(extent, crs=data_crs)

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 11}
gl.ylabel_style = {'size': 11}

# Anomaly fill (Seeded − Control)
diff = ivt_diff[idx]
im = ax.contourf(
    lon_grid, lat_grid, diff,
    levels=levels, cmap='RdBu_r', extend='both',
    transform=data_crs, alpha=0.9, zorder=2
)

# ========== PLOT ALL SEEDING LOCATIONS ==========
for i, location in enumerate(seeding_locations):
    lon_c = location['lon_center']
    lat_c = location['lat_center']
    radius = location['radius_km']

    # 1. Dashed circle outline (showing radius)
    plot_geodesic_circle(
        ax, lon_c, lat_c, radius,
        facecolor='none',
        edgecolor='black',
        linewidth=2.5,
        linestyle='--',
        zorder=5
    )

    # 2. Black star in the center
    ax.plot(
        lon_c, lat_c,
        marker='*',
        markersize=20,
        markerfacecolor='black',
        markeredgecolor='black',
        transform=data_crs,
        zorder=6
    )

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linewidth=2, linestyle='-', label='IVT Increase'),
    Line2D([0], [0], color='blue', linewidth=2, linestyle='-', label='IVT Decrease'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
           markersize=12, linestyle='None', label='Seeding Center'),
    Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Seeding Radius')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11,
         framealpha=0.95, edgecolor='black', fancybox=True)

# Title
max_abs = np.nanmax(np.abs(diff))
ax.set_title(
    f'Cloud Seeding Experiment: IVT Anomaly ({len(seeding_locations)} Seeding Regions)\n'
    f'Time: +{hours} hours | Max |Δ|: {max_abs:.0f} kg/(m·s)\n'
    f'(Red = increase, Blue = decrease)',
    fontsize=15, fontweight='bold', pad=12
)

# Colorbar
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
sm = ScalarMappable(norm=Normalize(vmin=-amax, vmax=amax), cmap='RdBu_r')
sm.set_array([])

cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7, extend='both')
cbar.set_label('IVT Anomaly [kg/(m·s)]  (Seeded − Control)', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=11)

plt.tight_layout()

# ========== SAVE FIGURE ==========
output_path = f'/scratch/moyanliu/Aurora Model/Perturbation/ivt_anomaly_t{hours:03d}hr.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {output_path}")

plt.show()
