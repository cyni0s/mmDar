"""Rasterize 3D lidar point clouds into polar occupancy grids.

Converts (N, 3) XYZ point clouds to (N_az, N_r) binary/soft occupancy
grids matching LISTA's angular grid convention:
    sin_theta[k] = -1 + 2*k/(N_az - 1), k = 0..N_az-1
    range[j] = j * r_max / (N_r - 1), j = 0..N_r-1
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def rasterize_to_polar(
    pts: np.ndarray,
    N_az: int = 256,
    N_r: int = 512,
    r_max: float = 10.8,
    sigma: float = 0.0,
) -> np.ndarray:
    """Convert XYZ point cloud to polar occupancy grid.

    Args:
        pts:   (N, 3) float32 point cloud [x, y, z]. z is ignored.
        N_az:  Number of azimuth bins (default 256, matches LISTA).
        N_r:   Number of range bins (default 512).
        r_max: Maximum range in meters (default 10.8).
        sigma: Gaussian softening sigma in bins (0 = hard binary).

    Returns:
        (N_az, N_r) float32 occupancy grid, values in [0, 1].
    """
    occ = np.zeros((N_az, N_r), dtype=np.float32)
    if len(pts) == 0:
        return occ

    x, y = pts[:, 0], pts[:, 1]
    r = np.sqrt(x**2 + y**2)
    sin_theta = np.zeros_like(r)
    np.divide(y, r, out=sin_theta, where=(r > 1e-8))

    mask = (r > 0.01) & (r <= r_max) & (x > 0) & (np.abs(sin_theta) <= 1.0)
    r = r[mask]
    sin_theta = sin_theta[mask]

    if len(r) == 0:
        return occ

    az_bins = np.round((sin_theta + 1.0) * (N_az - 1) / 2.0).astype(int)
    r_bins = np.round(r / r_max * (N_r - 1)).astype(int)
    az_bins = np.clip(az_bins, 0, N_az - 1)
    r_bins = np.clip(r_bins, 0, N_r - 1)

    occ[az_bins, r_bins] = 1.0

    if sigma > 0:
        occ = gaussian_filter(occ, sigma=sigma)
        occ = np.clip(occ / max(occ.max(), 1e-8), 0.0, 1.0)

    return occ


def rasterize_trajectory(lidar_pt_path, output_path, N_az=256, N_r=512, r_max=10.8, sigma=0.5):
    """Rasterize all frames in a lidar .pt to occupancy.
    Returns number of frames processed."""
    import torch
    lidar = torch.load(lidar_pt_path, weights_only=True).numpy()
    N = lidar.shape[0]
    occ_list = []
    for i in range(N):
        occ_list.append(rasterize_to_polar(lidar[i], N_az, N_r, r_max, sigma))
    occ = np.stack(occ_list)
    torch.save(torch.from_numpy(occ), output_path)
    return N


if __name__ == "__main__":
    import argparse, glob, os
    import torch
    parser = argparse.ArgumentParser(description="Rasterize lidar .pt to occupancy")
    parser.add_argument("--processed-dir", default="v2/data/processed")
    parser.add_argument("--sigma", type=float, default=0.5)
    args = parser.parse_args()
    lidar_files = sorted(glob.glob(os.path.join(args.processed_dir, "lidar_*.pt")))
    print(f"Found {len(lidar_files)} lidar .pt files")
    for lf in lidar_files:
        traj_id = os.path.basename(lf).replace("lidar_", "").replace(".pt", "")
        out_path = os.path.join(args.processed_dir, f"occ_{traj_id}.pt")
        if os.path.exists(out_path):
            print(f"  occ_{traj_id}.pt exists, skipping")
            continue
        n = rasterize_trajectory(lf, out_path, sigma=args.sigma)
        print(f"  occ_{traj_id}.pt: {n} frames rasterized")
