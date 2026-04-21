from .defaults import DEFAULT_SEEDING_PARAMS
from .mask import create_seeding_mask
from .physics import apply_physically_consistent_cloud_seeding


def apply_seeding(batch, seeding_locations, seeding_params=None):
    """Single high-level entry point for cloud seeding perturbation.

    Args:
        batch: Aurora Batch object (will be mutated in-place).
        seeding_locations: list of dicts, each with keys
            'lat_center', 'lon_center', 'radius_km'.
        seeding_params: optional dict of overrides merged with
            DEFAULT_SEEDING_PARAMS.

    Returns:
        (delta_T, delta_q, diagnostics)
    """
    merged_params = {**DEFAULT_SEEDING_PARAMS, **(seeding_params or {})}
    mask = create_seeding_mask(seeding_locations, batch.metadata.lat, batch.metadata.lon)
    return apply_physically_consistent_cloud_seeding(batch, mask, merged_params)
