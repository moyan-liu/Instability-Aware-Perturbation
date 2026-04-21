"""GPU-accelerated trajectory integrator using 4th-order Runge-Kutta."""

import torch


class GPUTrajectoryIntegrator:
    """
    GPU-accelerated trajectory integrator using 4th-order Runge-Kutta.
    Integrates all particle trajectories in parallel.
    """

    def __init__(self, wind_interpolator, dt=1.5):
        """
        Initialize the trajectory integrator.

        Args:
            wind_interpolator: GPUWindInterpolator instance
            dt: Time step in hours (default 1.5 hours)
        """
        self.wind_interp = wind_interpolator
        self.dt = dt
        self.device = wind_interpolator.device
        self.dtype = wind_interpolator.dtype

    def integrate(self, initial_lon, initial_lat, t_start, t_end):
        """
        Integrate trajectories from t_start to t_end using RK4.

        Args:
            initial_lon: [N] tensor of initial longitudes
            initial_lat: [N] tensor of initial latitudes
            t_start: Start time in hours
            t_end: End time in hours

        Returns:
            final_lon, final_lat: [N] tensors of final positions
        """
        # Current positions
        lon = initial_lon.clone()
        lat = initial_lat.clone()

        # Current time
        t = t_start

        # Determine direction (forward or backward integration)
        direction = 1.0 if t_end > t_start else -1.0
        dt = direction * abs(self.dt)

        # Number of steps
        n_steps = int(abs(t_end - t_start) / abs(self.dt))

        # RK4 integration
        for step in range(n_steps):
            # Ensure we don't overshoot
            if direction > 0 and t + dt > t_end:
                dt = t_end - t
            elif direction < 0 and t + dt < t_end:
                dt = t_end - t

            # Create time tensors
            t_tensor = torch.full_like(lon, t)
            t_half = torch.full_like(lon, t + 0.5 * dt)
            t_full = torch.full_like(lon, t + dt)

            # RK4 stages
            # k1
            k1_lon, k1_lat = self.wind_interp.velocity_degrees_per_hour(t_tensor, lon, lat)

            # k2
            lon_k2 = lon + 0.5 * dt * k1_lon
            lat_k2 = lat + 0.5 * dt * k1_lat
            k2_lon, k2_lat = self.wind_interp.velocity_degrees_per_hour(t_half, lon_k2, lat_k2)

            # k3
            lon_k3 = lon + 0.5 * dt * k2_lon
            lat_k3 = lat + 0.5 * dt * k2_lat
            k3_lon, k3_lat = self.wind_interp.velocity_degrees_per_hour(t_half, lon_k3, lat_k3)

            # k4
            lon_k4 = lon + dt * k3_lon
            lat_k4 = lat + dt * k3_lat
            k4_lon, k4_lat = self.wind_interp.velocity_degrees_per_hour(t_full, lon_k4, lat_k4)

            # Update positions
            lon = lon + (dt / 6.0) * (k1_lon + 2 * k2_lon + 2 * k3_lon + k4_lon)
            lat = lat + (dt / 6.0) * (k1_lat + 2 * k2_lat + 2 * k3_lat + k4_lat)

            # Do NOT wrap longitude here - the interpolator handles wrapping
            # internally via lon % 360. Wrapping here would create discontinuities
            # in the flow map that produce extreme FTLE values at the 0/360 boundary.

            # Clamp latitude
            lat = lat.clamp(-90, 90)

            # Update time
            t = t + dt

        return lon, lat
