"""Metrics for evaluating perturbation impact on California precipitation and IVT."""

import numpy as np


def compute_ca_precip_at_landfall(precip_list, landfall_steps,
                                 lats, lons, ca_mask_fn):
    """Sum precipitation over California during the landfall window.

    Args:
        precip_list: list of 2-D numpy precipitation arrays (one per step)
        landfall_steps: list of step indices in the landfall window
        lats: 1-D numpy array of latitudes
        lons: 1-D numpy array of longitudes
        ca_mask_fn: callable(lats, lons) -> 2-D boolean mask for California

    Returns:
        Total precipitation (float) summed over CA grid cells in the window.
    """
    ca_mask = ca_mask_fn(lats, lons)
    total = 0.0
    for s in landfall_steps:
        if s < len(precip_list):
            total += float(precip_list[s][ca_mask].sum())
    return total


def compute_ca_ivt_at_event(ivt_list, event_step, lats, lons, ca_mask_fn):
    """Mean IVT over California at the event step.

    Args:
        ivt_list: list of 2-D numpy IVT magnitude arrays (one per step)
        event_step: step index for the AR event
        lats: 1-D numpy array of latitudes
        lons: 1-D numpy array of longitudes
        ca_mask_fn: callable(lats, lons) -> 2-D boolean mask for California

    Returns:
        Mean IVT (float) over CA grid cells at the event step.
    """
    if event_step >= len(ivt_list):
        event_step = len(ivt_list) - 1
    ca_mask = ca_mask_fn(lats, lons)
    return float(ivt_list[event_step][ca_mask].mean())


def compute_ca_ivt_window(ivt_list, window_steps, lats, lons, ca_mask_fn):
    """Mean IVT over California averaged across a landfall window of steps.

    Args:
        ivt_list: list of 2-D numpy IVT magnitude arrays (one per step)
        window_steps: list of step indices in the landfall window
        lats: 1-D numpy array of latitudes
        lons: 1-D numpy array of longitudes
        ca_mask_fn: callable(lats, lons) -> 2-D boolean mask for California

    Returns:
        Mean IVT (float) over CA grid cells, averaged across all valid window steps.
    """
    ca_mask = ca_mask_fn(lats, lons)
    valid_steps = [s for s in window_steps if s < len(ivt_list)]
    if not valid_steps:
        return 0.0
    return float(np.mean([ivt_list[s][ca_mask].mean() for s in valid_steps]))


def compute_reduction(ctrl_value, test_value):
    """Compute fractional reduction relative to control.

    Returns:
        reduction: float in [0, 1] range if test < ctrl (positive = good).
    """
    if ctrl_value > 0:
        return (ctrl_value - test_value) / ctrl_value
    return 0.0
