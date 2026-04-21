"""Aurora rollout functions: control forecast and perturbed forecast."""

import dataclasses
import torch
import numpy as np


def rollout_store_velocities(model, decoder, batch, steps, transform_data_fn):
    """Rollout Aurora storing u, v, q velocity fields at each step.

    Args:
        model: Aurora model
        decoder: Decoder model (MLPDecoderLite)
        batch: Initial Aurora Batch
        steps: Number of 6-hour forecast steps
        transform_data_fn: Function to inverse-transform precipitation

    Returns:
        preds: list of Batch predictions (on CPU)
        precip_preds: list of precipitation arrays
        velocity_fields: list of dicts {'u', 'v', 'q'} as numpy [levels, lat, lon]
    """
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    preds = []
    precip_preds = []
    velocity_fields = []

    for i in range(steps):
        with torch.no_grad():
            pred, latent_decoder = model.forward(batch)

            latent_clone = latent_decoder.detach().clone()
            decoder_output = decoder.forward(
                latent_clone,
                batch.metadata.lat,
                batch.metadata.lon
            )

            precip = transform_data_fn(
                decoder_output['tp_mswep'].cpu().numpy().squeeze(),
                'tp_mswep',
                direct=False
            )

        vel = {
            'u': pred.atmos_vars['u'][0, 0].cpu().numpy(),
            'v': pred.atmos_vars['v'][0, 0].cpu().numpy(),
            'q': pred.atmos_vars['q'][0, 0].cpu().numpy(),
        }
        velocity_fields.append(vel)

        preds.append(pred.to("cpu"))
        precip_preds.append(precip)

        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )

    return preds, precip_preds, velocity_fields


def rollout_with_perturbation(model, decoder, batch, steps,
                              perturbation_schedule, seeding_params,
                              transform_data_fn,
                              create_seeding_mask_fn,
                              apply_seeding_fn):
    """Rollout with cloud seeding perturbations at specified steps.

    Args:
        model: Aurora model
        decoder: Decoder model
        batch: Initial Aurora Batch
        steps: Number of forecast steps
        perturbation_schedule: list of {'step', 'lat', 'lon', 'radius_km'}
        seeding_params: dict of cloud seeding parameters
        transform_data_fn: Precipitation inverse transform
        create_seeding_mask_fn: Function to create spatial mask
        apply_seeding_fn: Function to apply cloud seeding physics

    Returns:
        preds, precip_preds, velocity_fields
    """
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    # Organize perturbations by step
    pert_by_step = {}
    for p_info in perturbation_schedule:
        s = p_info['step']
        if s not in pert_by_step:
            pert_by_step[s] = []
        pert_by_step[s].append(p_info)

    preds = []
    precip_preds = []
    velocity_fields = []

    for i in range(steps):
        with torch.no_grad():
            pred, latent_decoder = model.forward(batch)

            latent_clone = latent_decoder.detach().clone()
            decoder_output = decoder.forward(
                latent_clone,
                batch.metadata.lat,
                batch.metadata.lon
            )

            precip = transform_data_fn(
                decoder_output['tp_mswep'].cpu().numpy().squeeze(),
                'tp_mswep',
                direct=False
            )

        vel = {
            'u': pred.atmos_vars['u'][0, 0].cpu().numpy(),
            'v': pred.atmos_vars['v'][0, 0].cpu().numpy(),
            'q': pred.atmos_vars['q'][0, 0].cpu().numpy(),
        }
        velocity_fields.append(vel)

        preds.append(pred.to("cpu"))
        precip_preds.append(precip)

        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )

        # Apply perturbation if scheduled
        if i in pert_by_step:
            seeding_locs = [{
                'lat_center': p_info['lat'],
                'lon_center': p_info['lon'],
                'radius_km': p_info.get('radius_km', 300)
            } for p_info in pert_by_step[i]]

            mask = create_seeding_mask_fn(
                seeding_locs,
                batch.metadata.lat,
                batch.metadata.lon
            )

            apply_seeding_fn(batch, mask, seeding_params)

            print(f"    Applied perturbation at step {i} (+{(i+1)*6}hr): "
                  f"{len(seeding_locs)} site(s)")

    return preds, precip_preds, velocity_fields


def rollout_store_velocities_ivt(model, batch, steps, pressure_levels, ivt_fn):
    """Rollout Aurora without decoder, computing IVT at each step.

    Args:
        model: Aurora model (no decoder needed)
        batch: Initial Aurora Batch
        steps: Number of 6-hour forecast steps
        pressure_levels: Tensor or list of pressure levels (hPa)
        ivt_fn: callable(u, v, q, pressure_levels) -> (ivt_u, ivt_v, ivt_mag)

    Returns:
        preds: list of Batch predictions (on CPU)
        ivt_preds: list of IVT magnitude arrays (lat x lon), numpy
        velocity_fields: list of dicts {'u', 'v', 'q'} as numpy [levels, lat, lon]
    """
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    preds = []
    ivt_preds = []
    velocity_fields = []

    for i in range(steps):
        with torch.no_grad():
            pred = model.forward(batch)

        u = pred.atmos_vars['u'][:, 0]
        v = pred.atmos_vars['v'][:, 0]
        q = pred.atmos_vars['q'][:, 0]
        _, _, ivt_mag = ivt_fn(u, v, q, pressure_levels)
        ivt_preds.append(ivt_mag[0].cpu().numpy())

        vel = {
            'u': pred.atmos_vars['u'][0, 0].cpu().numpy(),
            'v': pred.atmos_vars['v'][0, 0].cpu().numpy(),
            'q': pred.atmos_vars['q'][0, 0].cpu().numpy(),
        }
        velocity_fields.append(vel)
        preds.append(pred.to("cpu"))

        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )

    return preds, ivt_preds, velocity_fields


def rollout_with_perturbation_ivt(model, batch, steps,
                                   perturbation_schedule, seeding_params,
                                   create_seeding_mask_fn, apply_seeding_fn,
                                   pressure_levels, ivt_fn):
    """Rollout with cloud seeding perturbations, without decoder, computing IVT.

    Args:
        model: Aurora model (no decoder needed)
        batch: Initial Aurora Batch
        steps: Number of forecast steps
        perturbation_schedule: list of {'step', 'lat', 'lon', 'radius_km'}
        seeding_params: dict of cloud seeding parameters
        create_seeding_mask_fn: Function to create spatial mask
        apply_seeding_fn: Function to apply cloud seeding physics
        pressure_levels: Tensor or list of pressure levels (hPa)
        ivt_fn: callable(u, v, q, pressure_levels) -> (ivt_u, ivt_v, ivt_mag)

    Returns:
        preds, ivt_preds, velocity_fields
    """
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    pert_by_step = {}
    for p_info in perturbation_schedule:
        s = p_info['step']
        if s not in pert_by_step:
            pert_by_step[s] = []
        pert_by_step[s].append(p_info)

    preds = []
    ivt_preds = []
    velocity_fields = []

    for i in range(steps):
        with torch.no_grad():
            pred = model.forward(batch)

        u = pred.atmos_vars['u'][:, 0]
        v = pred.atmos_vars['v'][:, 0]
        q = pred.atmos_vars['q'][:, 0]
        _, _, ivt_mag = ivt_fn(u, v, q, pressure_levels)
        ivt_preds.append(ivt_mag[0].cpu().numpy())

        vel = {
            'u': pred.atmos_vars['u'][0, 0].cpu().numpy(),
            'v': pred.atmos_vars['v'][0, 0].cpu().numpy(),
            'q': pred.atmos_vars['q'][0, 0].cpu().numpy(),
        }
        velocity_fields.append(vel)
        preds.append(pred.to("cpu"))

        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )

        if i in pert_by_step:
            seeding_locs = [{
                'lat_center': p_info['lat'],
                'lon_center': p_info['lon'],
                'radius_km': p_info.get('radius_km', 300)
            } for p_info in pert_by_step[i]]

            mask = create_seeding_mask_fn(
                seeding_locs,
                batch.metadata.lat,
                batch.metadata.lon
            )
            apply_seeding_fn(batch, mask, seeding_params)
            print(f"    Applied perturbation at step {i} (+{(i+1)*6}hr): "
                  f"{len(seeding_locs)} site(s)")

    return preds, ivt_preds, velocity_fields
