"""FTLE-guided perturbation site selection for atmospheric river modification."""

from .defaults import DEFAULT_CONFIG
from .ftle import compute_ftle_and_catchment
from .scoring import score_perturbation_candidates
from .metrics import (
    compute_ca_precip_at_landfall,
    compute_ca_ivt_at_event,
    compute_ca_ivt_window,
    compute_reduction,
)
from .rollout import (
    rollout_store_velocities,
    rollout_with_perturbation,
    rollout_store_velocities_ivt,
    rollout_with_perturbation_ivt,
)

__all__ = [
    "DEFAULT_CONFIG",
    "compute_ftle_and_catchment",
    "score_perturbation_candidates",
    "compute_ca_precip_at_landfall",
    "compute_ca_ivt_at_event",
    "compute_ca_ivt_window",
    "compute_reduction",
    "rollout_store_velocities",
    "rollout_with_perturbation",
    "rollout_store_velocities_ivt",
    "rollout_with_perturbation_ivt",
]
