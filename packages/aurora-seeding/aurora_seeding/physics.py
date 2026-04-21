import torch


def calculate_q_sat(T, P):

    # Magnus formula for saturation vapor pressure (hPa)
    T_celsius = T - 273.15
    e_s = 6.112 * torch.exp((17.67 * T_celsius) / (T_celsius + 243.5))  # hPa

    # Convert to Pa
    e_s_Pa = e_s * 100

    # Saturation specific humidity
    epsilon = 0.622
    q_sat = epsilon * e_s_Pa / (P - 0.378 * e_s_Pa)

    return q_sat


def apply_physically_consistent_cloud_seeding(batch, seeding_mask_spatial, seeding_params):

    # ========== PHYSICAL CONSTANTS ==========
    L_f = 334000    # J/kg (latent heat of fusion)
    L_v = 2500000   # J/kg (latent heat of vaporization)
    L_d = L_v + L_f # J/kg (latent heat of DEPOSITION - vapor to ice)
    C_p = 1004      # J/(kg·K) (specific heat of air at constant pressure)

    # ========== PARAMETERS ==========
    layer_indices = [batch.metadata.atmos_levels.index(lev)
                     for lev in seeding_params['layers_mb']]
    pressure_levels_hPa = list(batch.metadata.atmos_levels)

    device = batch.atmos_vars["t"].device
    mask_torch = torch.from_numpy(seeding_mask_spatial).float().to(device)

    delta_T_applied = torch.zeros_like(batch.atmos_vars["t"][0, 1])
    delta_q_applied = torch.zeros_like(batch.atmos_vars["q"][0, 1])

    diagnostics = {
        'levels': [],
        'RH_before': [],
        'RH_after': [],
        'q_frozen_mean': [],
        'q_removed_mean': [],
        'delta_T_mean': [],
        'q_precipitated_mean': [],
        'energy_released_mean': [],
        'warnings': []
    }

    for level_idx in layer_indices:
        P_hPa = pressure_levels_hPa[level_idx]

        # ========== INITIAL STATE ==========
        T_old = batch.atmos_vars["t"][0, 1, level_idx].clone()
        q_old = batch.atmos_vars["q"][0, 1, level_idx].clone()
        P_Pa = P_hPa * 100  # hPa -> Pa

        # Saturation & RH
        q_sat_old = calculate_q_sat(T_old, P_Pa)
        RH_old_raw = q_old / (q_sat_old + 1e-10)
        RH_old = RH_old_raw.clamp(min=0.0)  # allow >1 for supersat

        # Parameters
        min_RH = seeding_params.get('min_RH', 0.8)
        freeze_efficiency = seeding_params.get('freeze_efficiency', 0.30)
        max_removal_fraction = seeding_params.get('max_removal_fraction', 0.80)
        fallout_fraction = seeding_params.get('fallout_fraction', 0.40)

        # Combined mask: spatial AND moist
        active_mask = ((mask_torch > 0) & (RH_old > min_RH)).float()
        seeded_region = active_mask > 0   # boolean mask

        # Potential freezing only where active
        q_frozen_potential = q_old * freeze_efficiency * active_mask

        # Moisture limit (never remove more than a fraction of q)
        q_frozen = torch.minimum(q_frozen_potential, q_old * max_removal_fraction)
        q_removed = q_frozen
        q_precip = q_frozen * fallout_fraction

        # Moisture-limited warning (use only cells that *could* have been seeded)
        if seeded_region.any():
            moisture_limited = (q_frozen[seeded_region] < q_frozen_potential[seeded_region]).any()
            if moisture_limited and level_idx == layer_indices[0]:
                pct_limited = ((q_frozen[seeded_region] < q_frozen_potential[seeded_region]).sum() /
                               seeded_region.sum() * 100).item()
                diagnostics['warnings'].append(
                    f"{pct_limited:.1f}% of seeded cells were moisture-limited "
                    f"(too dry for full {freeze_efficiency*100:.0f}% efficiency)"
                )

        # ========== LATENT HEAT RELEASE ==========
        delta_E = L_d * q_frozen
        delta_T = delta_E / C_p

        # ========== APPLY CHANGES ==========
        T_new = T_old + delta_T
        q_new = q_old - q_removed

        # ========== THERMODYNAMIC CONSISTENCY CHECK (SEEDED REGION ONLY) ==========
        q_sat_new = calculate_q_sat(T_new, P_Pa)
        RH_new = q_new / (q_sat_new + 1e-10)

        if seeded_region.any():
            q_new_seed = q_new[seeded_region]
            RH_new_seed = RH_new[seeded_region]
            RH_old_seed = RH_old[seeded_region]

            if (q_new_seed < 0).any():
                min_q = q_new_seed.min().item()
                diagnostics['warnings'].append(
                    f"Level {P_hPa:.0f} hPa: Negative q in seeded region! "
                    f"Min q = {min_q:.6f} kg/kg"
                )

            if (RH_new_seed > 1.5).any():
                max_rh = RH_new_seed.max().item()
                diagnostics['warnings'].append(
                    f"Level {P_hPa:.0f} hPa: High supersaturation in seeded region! "
                    f"Max RH = {max_rh*100:.1f}%"
                )

            if (RH_new_seed < 0.01).any() and (RH_old_seed.mean() > 0.1):
                min_rh = RH_new_seed.min().item()
                diagnostics['warnings'].append(
                    f"Level {P_hPa:.0f} hPa: Created very dry air in seeded region! "
                    f"Min RH = {min_rh*100:.1f}%"
                )

            RH_change_ratio = (RH_new_seed.mean() /
                               (RH_old_seed.mean() + 1e-10)).item()
            if RH_change_ratio < 0.2:
                diagnostics['warnings'].append(
                    f"Level {P_hPa:.0f} hPa: RH in seeded region dropped "
                    f"to {RH_change_ratio*100:.1f}% of original. "
                    f"Consider reducing freeze_efficiency or raising min_RH."
                )

        # ========== COMMIT CHANGES (ONLY timestep 1) ==========
        perturb_mode = seeding_params.get('perturb_mode', 'both')
        if perturb_mode in ('both', 'T_only'):
            batch.atmos_vars["t"][0, 1, level_idx] = T_new
        if perturb_mode in ('both', 'q_only'):
            batch.atmos_vars["q"][0, 1, level_idx] = q_new

        delta_T_applied[level_idx] = delta_T
        delta_q_applied[level_idx] = -q_removed

        # ========== STORE DIAGNOSTICS ==========
        diagnostics['levels'].append(P_hPa)

        if seeded_region.any():
            diagnostics['RH_before'].append(RH_old[seeded_region].mean().item())
            diagnostics['RH_after'].append(RH_new[seeded_region].mean().item())
            diagnostics['q_frozen_mean'].append(q_frozen[seeded_region].mean().item())
            diagnostics['q_removed_mean'].append(q_removed[seeded_region].mean().item())
            diagnostics['delta_T_mean'].append(delta_T[seeded_region].mean().item())
            diagnostics['q_precipitated_mean'].append(q_precip[seeded_region].mean().item())
            diagnostics['energy_released_mean'].append(delta_E[seeded_region].mean().item())
        else:
            # No seeded cells at this level - fill with zeros or NaNs
            diagnostics['RH_before'].append(float('nan'))
            diagnostics['RH_after'].append(float('nan'))
            diagnostics['q_frozen_mean'].append(0.0)
            diagnostics['q_removed_mean'].append(0.0)
            diagnostics['delta_T_mean'].append(0.0)
            diagnostics['q_precipitated_mean'].append(0.0)
            diagnostics['energy_released_mean'].append(0.0)

        # ========== VERTICAL COUPLING ==========
        if seeding_params.get('vertical_coupling', True):
            coupling_factor = seeding_params.get('coupling_factor', 0.3)
            for offset in [-1, 1]:
                adj_idx = level_idx + offset
                if 0 <= adj_idx < len(batch.metadata.atmos_levels):
                    delta_T_adj = delta_T * coupling_factor
                    batch.atmos_vars["t"][0, 1, adj_idx] += delta_T_adj
                    delta_T_applied[adj_idx] += delta_T_adj

    return delta_T_applied, delta_q_applied, diagnostics
