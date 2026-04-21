"""Candidate perturbation site scoring using FTLE, IVT, jet stream, and catchment."""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.interpolate import RegularGridInterpolator

from .ftle import compute_ftle_and_catchment
from .defaults import DEFAULT_CONFIG


def _haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points."""
    R = 6371.0
    lat1_r, lat2_r = np.deg2rad(lat1), np.deg2rad(lat2)
    dlat = np.deg2rad(lat2 - lat1)
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def score_perturbation_candidates(ivt_fields, vel_fields, lats, lons,
                                  ca_boundary_fn,
                                  wind_level_idx, jet_level_idx,
                                  ftle_gpu=None,
                                  config=None,
                                  **overrides):
    """Score candidate perturbation sites using multi-criteria filtering.

    Candidates must satisfy ALL criteria:
      1. High FTLE at 850 hPa (flow sensitivity)
      2. Near AR (IVT >= threshold)
      3. Near catchment boundary (CA-bound flow edge)
      4. At 250 hPa jet flank (steering-level sensitivity)
      5. OUTSIDE the CA target region (must perturb upstream)

    Score = FTLE value (pure dynamical sensitivity).

    Args:
        ivt_fields: list of 2-D numpy IVT magnitude arrays, one per step
        vel_fields: list of dicts {'u', 'v', 'q'} as numpy [levels, lat, lon]
        lats: 1-D numpy array of latitudes
        lons: 1-D numpy array of longitudes
        ca_boundary_fn: callable(lon_array, lat_array) -> boolean mask
        wind_level_idx: pressure-level index for 850 hPa winds
        jet_level_idx: pressure-level index for 250 hPa winds
        ftle_gpu: the ``ftle_gpu`` module
        config: dict of configuration (defaults to DEFAULT_CONFIG)
        **overrides: any key in config can be overridden here, e.g.
            ``candidates_per_step=20, min_separation_km=300``

    Returns:
        candidates: globally sorted list of dicts
        all_ftle_fields: dict step -> 2-D FTLE array
        all_catchment_maps: dict step -> 2-D boolean catchment array
        all_metadata: dict step -> metadata dict
    """
    cfg = {**DEFAULT_CONFIG}
    if config is not None:
        cfg.update(config)
    cfg.update(overrides)

    ca_lat_range = cfg['ca_lat_range']
    ca_lon_range = cfg['ca_lon_range']
    grid_spacing = cfg['ftle_grid_spacing']
    ivt_threshold = cfg['ivt_threshold']
    distance_threshold_km = cfg['perturbation_distance_threshold']
    jet_flank_min = cfg['jet_flank_min']
    jet_flank_max = cfg['jet_flank_max']
    candidates_per_step = cfg['candidates_per_step']
    min_separation_km = cfg['min_separation_km']
    ftle_lon_range = cfg['ftle_lon_range']
    ftle_lat_range = cfg['ftle_lat_range']

    candidates = []
    all_ftle_fields = {}
    all_catchment_maps = {}
    all_metadata = {}

    max_step = min(len(vel_fields) - 3, len(vel_fields))

    for t in range(max_step):
        print(f"  Scoring step {t} (+{(t+1)*6}hr)...", end=" ")

        # --- 850 hPa FTLE + Catchment ---
        ftle_field, catchment_mask, meta = compute_ftle_and_catchment(
            vel_fields, lats, lons,
            ca_boundary_fn=ca_boundary_fn,
            t_start_step=t,
            wind_level_idx=wind_level_idx,
            grid_spacing=grid_spacing,
            lon_range=ftle_lon_range,
            lat_range=ftle_lat_range,
            ftle_gpu=ftle_gpu,
        )

        if ftle_field is None:
            print("skipped (insufficient integration time)")
            continue

        all_ftle_fields[t] = ftle_field
        all_catchment_maps[t] = catchment_mask
        all_metadata[t] = meta

        grid_lons = meta['grid_lons']
        grid_lats = meta['grid_lats']

        if not catchment_mask.any():
            print("no catchment region found")
            continue

        # --- Catchment boundary ---
        dilated = binary_dilation(catchment_mask, iterations=2)
        eroded = binary_erosion(catchment_mask, iterations=1)
        boundary = (dilated & ~eroded) if eroded.any() else dilated

        # --- IVT on FTLE grid ---
        ivt_interp = RegularGridInterpolator(
            (lats, lons), ivt_fields[t],
            method='linear', bounds_error=False, fill_value=0
        )
        lon_grid_2d, lat_grid_2d = np.meshgrid(grid_lons, grid_lats)
        ivt_on_grid = ivt_interp(
            np.column_stack([lat_grid_2d.ravel(), lon_grid_2d.ravel()])
        ).reshape(meta['n_lat'], meta['n_lon'])

        ar_mask = ivt_on_grid >= ivt_threshold

        # --- Near-boundary region ---
        approx_deg = distance_threshold_km / 100
        near_boundary = binary_dilation(
            boundary, iterations=max(1, int(approx_deg / grid_spacing) + 1)
        )

        # --- 250 hPa jet stream on FTLE grid ---
        u_jet = vel_fields[t]['u'][jet_level_idx]
        v_jet = vel_fields[t]['v'][jet_level_idx]
        wind_speed_jet = np.sqrt(u_jet ** 2 + v_jet ** 2)

        jet_interp = RegularGridInterpolator(
            (lats, lons), wind_speed_jet,
            method='linear', bounds_error=False, fill_value=0
        )
        jet_on_grid = jet_interp(
            np.column_stack([lat_grid_2d.ravel(), lon_grid_2d.ravel()])
        ).reshape(meta['n_lat'], meta['n_lon'])

        jet_flank_mask = (jet_on_grid >= jet_flank_min) & (jet_on_grid <= jet_flank_max)

        # --- Exclusion mask: outside CA target region ---
        inside_ca_mask = (
            (lat_grid_2d >= ca_lat_range[0]) & (lat_grid_2d <= ca_lat_range[1]) &
            (lon_grid_2d >= ca_lon_range[0]) & (lon_grid_2d <= ca_lon_range[1])
        )
        outside_ca_mask = ~inside_ca_mask

        # --- Combined scoring mask ---
        scoring_mask = ar_mask & near_boundary & jet_flank_mask & outside_ca_mask

        n_ar = ar_mask.sum()
        n_boundary = near_boundary.sum()
        n_jet = jet_flank_mask.sum()
        n_outside_ca = outside_ca_mask.sum()

        if not scoring_mask.any():
            # Fallback: try without jet constraint (but still outside CA)
            scoring_mask_no_jet = ar_mask & near_boundary & outside_ca_mask
            if scoring_mask_no_jet.any():
                print(f"no jet flank overlap (AR:{n_ar}, boundary:{n_boundary}, "
                      f"jet:{n_jet}, outside_CA:{n_outside_ca}, combined:0). "
                      f"Falling back to AR+boundary only (outside CA).")
                scoring_mask = scoring_mask_no_jet
            else:
                print(f"no candidates outside CA (AR:{n_ar}, boundary:{n_boundary}, jet:{n_jet})")
                continue

        # --- Find top K candidates with minimum separation ---
        ftle_scored = ftle_field * scoring_mask
        flat_scores = ftle_scored.ravel().copy()
        step_candidates = []

        for _ in range(candidates_per_step * 3):
            if len(step_candidates) >= candidates_per_step:
                break

            flat_idx = np.argmax(flat_scores)
            if flat_scores[flat_idx] <= 0:
                break

            row, col = np.unravel_index(flat_idx, ftle_field.shape)
            cand_lat = float(grid_lats[row])
            cand_lon = float(grid_lons[col])
            cand_ftle = float(ftle_field[row, col])
            cand_ivt = float(ivt_on_grid[row, col])
            cand_jet = float(jet_on_grid[row, col])

            # Exclusion: must be outside target region
            if (ca_lat_range[0] <= cand_lat <= ca_lat_range[1] and
                    ca_lon_range[0] <= cand_lon <= ca_lon_range[1]):
                flat_scores[flat_idx] = 0
                continue

            # Must be in jet flank
            if not (jet_flank_min <= cand_jet <= jet_flank_max):
                flat_scores[flat_idx] = 0
                continue

            # Score = pure FTLE
            cand_score = cand_ftle

            # Separation check
            too_close = False
            for existing in step_candidates:
                dist = _haversine_distance(cand_lat, cand_lon,
                                           existing['lat'], existing['lon'])
                if dist < min_separation_km:
                    too_close = True
                    break
            if too_close:
                flat_scores[flat_idx] = 0
                continue

            step_candidates.append({
                'step': t,
                'lat': cand_lat,
                'lon': cand_lon,
                'score': cand_score,
                'ftle_value': cand_ftle,
                'ivt_value': cand_ivt,
                'jet_speed': cand_jet,
                'jet_factor': 1.0,
            })

            # Zero out region around selected candidate
            mask_radius_idx = max(1, int(min_separation_km / (grid_spacing * 111)))
            r_start = max(0, row - mask_radius_idx)
            r_end = min(ftle_field.shape[0], row + mask_radius_idx + 1)
            c_start = max(0, col - mask_radius_idx)
            c_end = min(ftle_field.shape[1], col + mask_radius_idx + 1)
            flat_mask = np.zeros_like(ftle_scored, dtype=bool)
            flat_mask[r_start:r_end, c_start:c_end] = True
            flat_scores[flat_mask.ravel()] = 0

        candidates.extend(step_candidates)

        if step_candidates:
            best = step_candidates[0]
            print(f"{len(step_candidates)} candidates, "
                  f"best: score={best['score']:.4f}, "
                  f"FTLE={best['ftle_value']:.4f}, "
                  f"jet={best['jet_speed']:.0f}m/s, "
                  f"at ({best['lat']:.1f}N, {best['lon']:.1f}E)")
        else:
            print("no valid candidates found")

    # Global sort by score
    candidates.sort(key=lambda x: x['score'], reverse=True)

    print(f"\nTotal candidates: {len(candidates)}")
    if candidates:
        print(f"\nTop 10 globally (all OUTSIDE CA target region):")
        print(f"  {'#':<4} {'Step':<6} {'Lat':>6} {'Lon':>7} {'Score':>7} "
              f"{'FTLE':>7} {'IVT':>5} {'Jet':>5}")
        for i, c in enumerate(candidates[:10]):
            print(f"  {i+1:<4} {c['step']:<6} {c['lat']:>6.1f} {c['lon']:>7.1f} "
                  f"{c['score']:>7.3f} {c['ftle_value']:>7.4f} "
                  f"{c['ivt_value']:>5.0f} {c['jet_speed']:>5.0f}")

    from collections import Counter
    step_counts = Counter(c['step'] for c in candidates)
    print(f"\nCandidates per step: {dict(sorted(step_counts.items()))}")

    return candidates, all_ftle_fields, all_catchment_maps, all_metadata
