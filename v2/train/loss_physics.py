"""Physics-informed radar losses: differentiable soft-splatting + polar losses.

IWR1443 parameters: sigma_r=1.0 bins, sigma_u=14 bins (fixed in sin-theta space).
Grid: 256 azimuth x 512 range, [sin_theta in [-1,1]] x [range in [0, 10.8m]]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_gaussian_kernel_1d(sigma: float, truncate: float = 3.0) -> torch.Tensor:
    """1D Gaussian kernel normalized to sum=1."""
    radius = int(math.ceil(sigma * truncate))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


class SoftSplat(nn.Module):
    """Differentiable soft-splatting: pts (B, N, 3) -> O (B, 1, H, W) in [0, 1).

    Pipeline: bilinear soft-bin -> separable Gaussian blur -> 1 - exp(-I)
    """

    def __init__(
        self,
        N_az: int = 256,
        N_r: int = 512,
        r_max: float = 10.8,
        sigma_r: float = 1.0,
        sigma_u: float = 14.0,
    ):
        super().__init__()
        self.N_az = N_az
        self.N_r = N_r
        self.r_max = r_max

        # Pre-compute 1D Gaussian kernels as conv2d weights
        kr = _make_gaussian_kernel_1d(sigma_r)
        ku = _make_gaussian_kernel_1d(sigma_u)
        # Shape for conv2d: (out_ch, in_ch, kH, kW)
        self.register_buffer("kernel_r", kr.view(1, 1, 1, -1))
        self.register_buffer("kernel_u", ku.view(1, 1, -1, 1))
        self.pad_r = kr.shape[0] // 2
        self.pad_u = ku.shape[0] // 2

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pts: (B, N, 3) point cloud in Cartesian (x=forward, y=left, z=up)

        Returns:
            occ: (B, 1, N_az, N_r) polar occupancy grid in [0, 1)
        """
        B = pts.shape[0]
        device = pts.device
        dtype = pts.dtype
        intensity = torch.zeros(B, self.N_az * self.N_r, device=device, dtype=dtype)

        if pts.shape[1] == 0:
            return intensity.view(B, 1, self.N_az, self.N_r)

        # Cartesian -> polar
        x, y = pts[..., 0], pts[..., 1]
        r = torch.sqrt(x ** 2 + y ** 2 + 1e-8)
        u = y / (r + 1e-8)  # sin(theta)

        # Validity mask: x > 0 (forward hemisphere), range within max, |sin_theta| <= 1
        valid = (x > 0.01) & (r <= self.r_max) & (u.abs() <= 1.0)

        # Continuous grid coordinates
        r_coord = r / self.r_max * (self.N_r - 1)     # [0, N_r-1]
        u_coord = (u + 1.0) * (self.N_az - 1) / 2.0  # [0, N_az-1]

        # Bilinear corners
        r_floor = r_coord.long().clamp(0, self.N_r - 2)
        u_floor = u_coord.long().clamp(0, self.N_az - 2)
        r_frac = (r_coord - r_floor.float()).clamp(0, 1)
        u_frac = (u_coord - u_floor.float()).clamp(0, 1)

        # Bilinear weights, zeroed for invalid points
        mask = valid.float()
        w00 = (1 - r_frac) * (1 - u_frac) * mask
        w01 = r_frac * (1 - u_frac) * mask
        w10 = (1 - r_frac) * u_frac * mask
        w11 = r_frac * u_frac * mask

        # Flat indices into (N_az * N_r) grid
        r_floor_p1 = (r_floor + 1).clamp(max=self.N_r - 1)
        u_floor_p1 = (u_floor + 1).clamp(max=self.N_az - 1)
        idx00 = u_floor * self.N_r + r_floor
        idx01 = u_floor * self.N_r + r_floor_p1
        idx10 = u_floor_p1 * self.N_r + r_floor
        idx11 = u_floor_p1 * self.N_r + r_floor_p1

        # Scatter-add per batch element (B typically <= 12, so loop is fine)
        for b in range(B):
            intensity[b].scatter_add_(0, idx00[b], w00[b])
            intensity[b].scatter_add_(0, idx01[b], w01[b])
            intensity[b].scatter_add_(0, idx10[b], w10[b])
            intensity[b].scatter_add_(0, idx11[b], w11[b])

        intensity = intensity.view(B, 1, self.N_az, self.N_r)

        # Separable Gaussian blur (PSF convolution)
        intensity = F.conv2d(intensity, self.kernel_u, padding=(self.pad_u, 0))
        intensity = F.conv2d(intensity, self.kernel_r, padding=(0, self.pad_r))

        # Bounded output: 1 - exp(-I) maps [0, inf) -> [0, 1)
        return 1.0 - torch.exp(-intensity)
