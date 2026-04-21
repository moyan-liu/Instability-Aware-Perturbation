"""GPU-accelerated FTLE calculation with vectorized gradient and eigenvalue computation."""

import torch


class GPUFTLECalculator:
    """
    GPU-accelerated FTLE calculation.
    Computes flow map gradient and FTLE field using vectorized operations.
    """

    def __init__(self, device, dtype=torch.float64):
        self.device = device
        self.dtype = dtype

    def compute_flow_map_gradient(self, initial_lon, initial_lat, final_lon, final_lat,
                                  grid_shape, grid_spacing):
        """
        Compute the flow map gradient using finite differences.
        Vectorized implementation - no loops.

        Args:
            initial_lon, initial_lat: [N] initial positions (flattened grid)
            final_lon, final_lat: [N] final positions
            grid_shape: (n_lat, n_lon) tuple
            grid_spacing: spacing in degrees

        Returns:
            grad_phi: [n_lat, n_lon, 2, 2] gradient tensor
        """
        n_lat, n_lon = grid_shape

        # Reshape to grid
        final_lon_grid = final_lon.reshape(n_lat, n_lon)
        final_lat_grid = final_lat.reshape(n_lat, n_lon)

        # Initialize gradient tensor
        grad_phi = torch.zeros(n_lat, n_lon, 2, 2, device=self.device, dtype=self.dtype)

        # Central differences for interior points (vectorized)
        # d(final_lon)/d(init_lon) - varies along longitude direction (j)
        grad_phi[1:-1, 1:-1, 0, 0] = (final_lon_grid[1:-1, 2:] - final_lon_grid[1:-1, :-2]) / (2 * grid_spacing)

        # d(final_lon)/d(init_lat) - varies along latitude direction (i)
        grad_phi[1:-1, 1:-1, 0, 1] = (final_lon_grid[2:, 1:-1] - final_lon_grid[:-2, 1:-1]) / (2 * grid_spacing)

        # d(final_lat)/d(init_lon)
        grad_phi[1:-1, 1:-1, 1, 0] = (final_lat_grid[1:-1, 2:] - final_lat_grid[1:-1, :-2]) / (2 * grid_spacing)

        # d(final_lat)/d(init_lat)
        grad_phi[1:-1, 1:-1, 1, 1] = (final_lat_grid[2:, 1:-1] - final_lat_grid[:-2, 1:-1]) / (2 * grid_spacing)

        # Boundary handling with forward/backward differences
        # Left boundary (j=0)
        grad_phi[:, 0, 0, 0] = (final_lon_grid[:, 1] - final_lon_grid[:, 0]) / grid_spacing
        grad_phi[:, 0, 1, 0] = (final_lat_grid[:, 1] - final_lat_grid[:, 0]) / grid_spacing

        # Right boundary (j=-1)
        grad_phi[:, -1, 0, 0] = (final_lon_grid[:, -1] - final_lon_grid[:, -2]) / grid_spacing
        grad_phi[:, -1, 1, 0] = (final_lat_grid[:, -1] - final_lat_grid[:, -2]) / grid_spacing

        # Bottom boundary (i=0)
        grad_phi[0, :, 0, 1] = (final_lon_grid[1, :] - final_lon_grid[0, :]) / grid_spacing
        grad_phi[0, :, 1, 1] = (final_lat_grid[1, :] - final_lat_grid[0, :]) / grid_spacing

        # Top boundary (i=-1)
        grad_phi[-1, :, 0, 1] = (final_lon_grid[-1, :] - final_lon_grid[-2, :]) / grid_spacing
        grad_phi[-1, :, 1, 1] = (final_lat_grid[-1, :] - final_lat_grid[-2, :]) / grid_spacing

        return grad_phi

    def compute_cauchy_green_tensor(self, grad_phi, lats_grid):
        """
        Compute Cauchy-Green strain tensor with spherical metric.
        Vectorized - no loops.

        Args:
            grad_phi: [n_lat, n_lon, 2, 2] gradient tensor
            lats_grid: [n_lat] latitude array

        Returns:
            C: [n_lat, n_lon, 2, 2] Cauchy-Green tensor
        """
        n_lat, n_lon = grad_phi.shape[:2]

        # Convert lats to tensor if needed
        if not torch.is_tensor(lats_grid):
            lats_grid = torch.tensor(lats_grid, device=self.device, dtype=self.dtype)

        lat_rad = lats_grid * (torch.pi / 180)  # [n_lat]
        cos_lat_sq = torch.cos(lat_rad) ** 2  # [n_lat]

        # Metric tensor G (diagonal for spherical coordinates)
        # G = diag([cos^2(lat), 1])
        G = torch.zeros(n_lat, n_lon, 2, 2, device=self.device, dtype=self.dtype)
        G[:, :, 0, 0] = cos_lat_sq[:, None]  # Broadcast across lon
        G[:, :, 1, 1] = 1.0

        # Cauchy-Green tensor: C = (nabla Phi)^T @ G @ (nabla Phi)
        grad_phi_T = grad_phi.transpose(-2, -1)  # [n_lat, n_lon, 2, 2]

        # C = grad_phi_T @ G @ grad_phi
        temp = torch.matmul(G, grad_phi)  # [n_lat, n_lon, 2, 2]
        C = torch.matmul(grad_phi_T, temp)  # [n_lat, n_lon, 2, 2]

        return C

    def compute_ftle(self, grad_phi, lats_grid, tau_hours):
        """
        Compute FTLE field from flow map gradient.
        Uses batched eigenvalue computation.

        Args:
            grad_phi: [n_lat, n_lon, 2, 2] gradient tensor
            lats_grid: [n_lat] latitude array
            tau_hours: integration time in hours

        Returns:
            ftle: [n_lat, n_lon] FTLE field
        """
        n_lat, n_lon = grad_phi.shape[:2]

        # Compute Cauchy-Green tensor
        C = self.compute_cauchy_green_tensor(grad_phi, lats_grid)

        # Reshape for batched eigenvalue computation
        C_flat = C.reshape(-1, 2, 2)  # [n_lat * n_lon, 2, 2]

        # Batched eigenvalue computation
        eigenvalues = torch.linalg.eigvals(C_flat)  # [n_lat * n_lon, 2], complex

        # Get maximum eigenvalue (real part)
        max_eigenvalues = eigenvalues.real.max(dim=1)[0]  # [n_lat * n_lon]

        # Compute FTLE: (1/|tau|) * log(sqrt(lambda_max))
        # = (1/(2|tau|)) * log(lambda_max)
        tau_days = tau_hours / 24.0
        ftle_flat = torch.where(
            max_eigenvalues > 0,
            (1 / abs(tau_days)) * torch.log(torch.sqrt(max_eigenvalues.clamp(min=1e-10))),
            torch.zeros_like(max_eigenvalues),
        )

        # Reshape back to grid
        ftle = ftle_flat.reshape(n_lat, n_lon)

        return ftle
