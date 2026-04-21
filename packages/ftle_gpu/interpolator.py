"""GPU-accelerated wind field interpolator using trilinear interpolation."""

import torch


class GPUWindInterpolator:
    """
    GPU-accelerated wind field interpolator using trilinear interpolation.
    Matches scipy.RegularGridInterpolator behavior for exact results.
    """

    def __init__(self, u_wind, v_wind, times_hours, lats, lons, device, dtype=torch.float64):
        """
        Initialize the interpolator.

        Args:
            u_wind: [T, lat, lon] tensor of u-wind component
            v_wind: [T, lat, lon] tensor of v-wind component
            times_hours: [T] array of times in hours
            lats: [lat] array of latitudes (descending, 90 to -90)
            lons: [lon] array of longitudes (0 to 360)
            device: torch device
            dtype: torch dtype
        """
        self.device = device
        self.dtype = dtype

        # Store wind data
        self.u_wind = u_wind  # Already on GPU
        self.v_wind = v_wind

        # Store coordinate arrays
        self.times = torch.tensor(times_hours, device=device, dtype=dtype)
        self.lats = lats if torch.is_tensor(lats) else torch.tensor(lats, device=device, dtype=dtype)
        self.lons = lons if torch.is_tensor(lons) else torch.tensor(lons, device=device, dtype=dtype)

        # Grid parameters
        self.n_times = len(self.times)
        self.n_lats = len(self.lats)
        self.n_lons = len(self.lons)

        # Check if lats are descending (90 to -90) or ascending
        self.lats_descending = self.lats[0] > self.lats[-1]

        # Earth's radius in km
        self.R_earth = 6371.0

    def _trilinear_interp(self, data, t_idx, lat_idx, lon_idx, t_frac, lat_frac, lon_frac):
        """
        Perform trilinear interpolation.

        Args:
            data: [T, lat, lon] tensor
            t_idx, lat_idx, lon_idx: integer indices (lower corner)
            t_frac, lat_frac, lon_frac: fractional parts [0, 1]
        """
        # Clamp indices to valid range
        t0 = t_idx.clamp(0, self.n_times - 2)
        t1 = (t0 + 1).clamp(0, self.n_times - 1)

        lat0 = lat_idx.clamp(0, self.n_lats - 2)
        lat1 = (lat0 + 1).clamp(0, self.n_lats - 1)

        lon0 = lon_idx.clamp(0, self.n_lons - 2)
        lon1 = (lon0 + 1).clamp(0, self.n_lons - 1)

        # Get corner values - using advanced indexing
        c000 = data[t0, lat0, lon0]
        c001 = data[t0, lat0, lon1]
        c010 = data[t0, lat1, lon0]
        c011 = data[t0, lat1, lon1]
        c100 = data[t1, lat0, lon0]
        c101 = data[t1, lat0, lon1]
        c110 = data[t1, lat1, lon0]
        c111 = data[t1, lat1, lon1]

        # Trilinear interpolation
        # Interpolate along lon
        c00 = c000 * (1 - lon_frac) + c001 * lon_frac
        c01 = c010 * (1 - lon_frac) + c011 * lon_frac
        c10 = c100 * (1 - lon_frac) + c101 * lon_frac
        c11 = c110 * (1 - lon_frac) + c111 * lon_frac

        # Interpolate along lat
        c0 = c00 * (1 - lat_frac) + c01 * lat_frac
        c1 = c10 * (1 - lat_frac) + c11 * lat_frac

        # Interpolate along time
        result = c0 * (1 - t_frac) + c1 * t_frac

        return result

    def interpolate(self, t, lon, lat):
        """
        Interpolate wind at given positions.

        Args:
            t: [N] tensor of times in hours
            lon: [N] tensor of longitudes
            lat: [N] tensor of latitudes

        Returns:
            u, v: [N] tensors of interpolated wind components
        """
        # Clip latitude to valid range
        lat = lat.clamp(-90, 90)

        # Handle longitude wrapping (convert to 0-360 if needed)
        lon = lon % 360

        # Compute normalized coordinates
        # Time
        t_norm = (t - self.times[0]) / (self.times[-1] - self.times[0]) * (self.n_times - 1)
        t_idx = t_norm.long()
        t_frac = t_norm - t_idx.to(self.dtype)

        # Latitude (handle descending order)
        if self.lats_descending:
            lat_norm = (self.lats[0] - lat) / (self.lats[0] - self.lats[-1]) * (self.n_lats - 1)
        else:
            lat_norm = (lat - self.lats[0]) / (self.lats[-1] - self.lats[0]) * (self.n_lats - 1)
        lat_idx = lat_norm.long()
        lat_frac = lat_norm - lat_idx.to(self.dtype)

        # Longitude
        lon_norm = (lon - self.lons[0]) / (self.lons[-1] - self.lons[0]) * (self.n_lons - 1)
        lon_idx = lon_norm.long()
        lon_frac = lon_norm - lon_idx.to(self.dtype)

        # Interpolate
        u = self._trilinear_interp(self.u_wind, t_idx, lat_idx, lon_idx, t_frac, lat_frac, lon_frac)
        v = self._trilinear_interp(self.v_wind, t_idx, lat_idx, lon_idx, t_frac, lat_frac, lon_frac)

        # Set to zero outside valid time range
        outside_time = (t < self.times[0]) | (t > self.times[-1])
        u = torch.where(outside_time, torch.zeros_like(u), u)
        v = torch.where(outside_time, torch.zeros_like(v), v)

        return u, v

    def velocity_degrees_per_hour(self, t, lon, lat):
        """
        Compute velocity in degrees per hour for trajectory integration.

        Args:
            t: [N] tensor of times in hours
            lon: [N] tensor of longitudes
            lat: [N] tensor of latitudes

        Returns:
            dlon_dt, dlat_dt: [N] tensors of velocity in degrees/hour
        """
        u, v = self.interpolate(t, lon, lat)

        # Convert from m/s to degrees/hour
        # dlon/dt = u / (R * cos(lat)) * (180/pi) * 3600
        # dlat/dt = v / R * (180/pi) * 3600
        lat_rad = lat * (torch.pi / 180)
        cos_lat = torch.cos(lat_rad).clamp(min=0.01)  # Avoid division by zero at poles

        dlon_dt = (u / (self.R_earth * 1000 * cos_lat)) * (180 / torch.pi) * 3600
        dlat_dt = (v / (self.R_earth * 1000)) * (180 / torch.pi) * 3600

        return dlon_dt, dlat_dt
