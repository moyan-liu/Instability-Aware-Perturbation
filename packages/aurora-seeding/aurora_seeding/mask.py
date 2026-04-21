import numpy as np
import torch


def haversine_distance(lat1, lon1, lat2, lon2):

    R = 6371.0  # Earth radius in km

    # Convert to radians
    lat1_rad = np.deg2rad(lat1)
    lat2_rad = np.deg2rad(lat2)
    lon1_rad = np.deg2rad(lon1)
    lon2_rad = np.deg2rad(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    distance = R * c
    return distance


def create_seeding_mask(seeding_locations, lat_grid, lon_grid):

    # Convert torch tensors to numpy if needed
    if torch.is_tensor(lat_grid):
        lat_grid = lat_grid.cpu().numpy()
    if torch.is_tensor(lon_grid):
        lon_grid = lon_grid.cpu().numpy()

    # If 1D arrays, create 2D grids
    if lat_grid.ndim == 1 and lon_grid.ndim == 1:
        lon_grid_2d, lat_grid_2d = np.meshgrid(lon_grid, lat_grid)
    else:
        lat_grid_2d = lat_grid
        lon_grid_2d = lon_grid

    # Initialize empty mask
    combined_mask = np.zeros_like(lat_grid_2d, dtype=bool)

    # Loop through each seeding location
    for location in seeding_locations:
        lat_center = location['lat_center']
        lon_center = location['lon_center']
        radius_km = location['radius_km']

        # Calculate distance from this center to all grid points
        distances = haversine_distance(lat_center, lon_center, lat_grid_2d, lon_grid_2d)

        # Create circular mask for this location
        circle_mask = distances <= radius_km

        # Add to combined mask (logical OR - union of all circles)
        combined_mask = combined_mask | circle_mask

    return combined_mask
