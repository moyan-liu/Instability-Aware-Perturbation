# Default configuration for FTLE-guided perturbation pipeline.
# All values can be overridden at call time.

DEFAULT_CONFIG = {
    # --- CA target region (0-360 longitude format) ---
    'ca_lat_range': (32, 42),
    'ca_lon_range': (235, 245),

    # --- FTLE / flow sensitivity ---
    'wind_level': 850,            # hPa — FTLE computed at AR moisture transport level
    'ftle_grid_spacing': 0.75,    # degrees
    'ivt_threshold': 250,         # kg/(m·s) for AR detection

    # --- Jet stream (steering level) ---
    'jet_level': 250,             # hPa — upper-level jet for AR steering
    'jet_flank_min': 30,          # m/s — below this, too far from jet
    'jet_flank_max': 60,          # m/s — above this, inside jet core

    # --- Perturbation search ---
    'perturbation_distance_threshold': 300,  # km

    # --- Seeding ---
    'default_seeding_radius_km': 250,

    # --- Multi-site selection ---
    'candidates_per_step': 10,
    'min_separation_km': 500,     # km — avoid clustering
    'num_sites_initial': 5,
    'sites_to_add_per_round': 2,
    'max_rounds': 3,

    # --- AR event target (override per event) ---
    'event_step': 14,
    'landfall_window_steps': [13, 14, 15],
    'precipitation_reduction_target': 0.10,

    # --- FTLE domain ---
    'ftle_lon_range': (150, 260),
    'ftle_lat_range': (0, 70),
}
