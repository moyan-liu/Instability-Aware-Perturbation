"""Minimal usage example for aurora_seeding.

Assumes you already have an Aurora Batch object (e.g. ``batch_seeded``)
loaded on CUDA.  See the Aurora documentation or Version_1.ipynb for how
to construct one from ERA5 data.
"""

from aurora_seeding import apply_seeding

# --- 1. Minimal call: only locations required, defaults for everything else ---
# delta_T, delta_q, diagnostics = apply_seeding(
#     batch=batch_seeded,
#     seeding_locations=[
#         {'lat_center': 33.8, 'lon_center': 171,   'radius_km': 300},
#         {'lat_center': 31.5, 'lon_center': 160.5, 'radius_km': 300},
#     ],
# )

# --- 2. Override specific params; rest uses defaults ---
# delta_T, delta_q, diagnostics = apply_seeding(
#     batch=batch_seeded,
#     seeding_locations=[
#         {'lat_center': 33.8, 'lon_center': 171, 'radius_km': 300},
#     ],
#     seeding_params={
#         'freeze_efficiency': 0.30,
#         'layers_mb': [850, 700],
#     },
# )

# --- Quick import smoke-test (no GPU needed) ---
if __name__ == "__main__":
    from aurora_seeding import (
        apply_seeding,
        DEFAULT_SEEDING_PARAMS,
        create_seeding_mask,
        haversine_distance,
        apply_physically_consistent_cloud_seeding,
        calculate_q_sat,
    )
    print("All public symbols imported successfully.")
    print("Default params:", DEFAULT_SEEDING_PARAMS)
