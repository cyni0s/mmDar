"""Preprocess raw IQ through FFTBeamformer → log_power → polar angle-uniform grid.

Run inside Docker:
  docker compose run --rm mmdar python3 v2/data/preprocess_lista.py
"""
import os, sys, torch, numpy as np
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from v2.model.lista import FFTBeamformer
from v2.data.split import ALL_TRAJS
from eval.eval_pointcloud import RMAX, RBINS, ABINS

PROCESSED_DIR = 'v2/data/processed'
N_AZ_LISTA = 256


def reproject_azimuth(log_power: torch.Tensor) -> torch.Tensor:
    """Reproject from sin_theta-uniform (256 bins) to angle-uniform (512 bins).

    Args:
        log_power: (N, 256_az_sintheta, 512_range) float32

    Returns:
        (N, 512_az_angle, 512_range) float32
    """
    N, A_in, R = log_power.shape
    angles = torch.linspace(-90, 90, ABINS)
    sin_vals = torch.sin(angles * torch.pi / 180)
    grid_az = sin_vals.unsqueeze(1).expand(ABINS, R)
    grid_r = torch.linspace(-1, 1, R).unsqueeze(0).expand(ABINS, R)
    grid = torch.stack([grid_r, grid_az], dim=-1)
    grid = grid.unsqueeze(0).expand(N, -1, -1, -1)
    inp = log_power.unsqueeze(1)
    out = F.grid_sample(inp, grid.to(inp.device), mode='bilinear',
                        padding_mode='zeros', align_corners=True)
    return out.squeeze(1)


def rasterize_lidar_to_polar_grid(pts: np.ndarray) -> np.ndarray:
    """Rasterize (8192, 3) point cloud to (range, angle-uniform) polar grid (256r × 512az)."""
    r_grid = np.linspace(0, RMAX, RBINS)
    a_grid = np.linspace(-90, 90, ABINS)
    grid = np.zeros((RBINS, ABINS), dtype=np.uint8)
    if len(pts) == 0:
        return grid
    x, y = pts[:, 0], pts[:, 1]
    r = np.sqrt(x**2 + y**2)
    angle_deg = np.degrees(np.arctan2(y, x))
    mask = (r > 0.01) & (r <= RMAX) & (x > 0) & (np.abs(angle_deg) <= 90)
    r, angle_deg = r[mask], angle_deg[mask]
    if len(r) == 0:
        return grid
    row = np.clip(np.searchsorted(r_grid, r, side='left'), 0, RBINS - 1)
    col = np.clip(np.searchsorted(a_grid, angle_deg, side='left'), 0, ABINS - 1)
    grid[row, col] = 1
    return grid


def process_trajectory(tid: int, bf: FFTBeamformer, device: torch.device):
    radar_path = os.path.join(PROCESSED_DIR, f'radar_{tid}.pt')
    lidar_path = os.path.join(PROCESSED_DIR, f'lidar_{tid}.pt')
    if not os.path.exists(radar_path):
        print(f'  Skip {tid}: no radar file')
        return

    radar = torch.load(radar_path, weights_only=True)
    lidar = torch.load(lidar_path, weights_only=True).numpy()
    N = radar.shape[0]
    print(f'  Traj {tid}: {N} frames')

    CHUNK = 256
    logpow_list = []
    for s in range(0, N, CHUNK):
        e = min(s + CHUNK, N)
        with torch.no_grad():
            spec = bf(radar[s:e].to(device))
        lp = torch.log(spec.real**2 + spec.imag**2 + 1e-6).cpu()
        logpow_list.append(lp)
    log_power = torch.cat(logpow_list, dim=0)

    reprojected = reproject_azimuth(log_power)
    downsampled = reprojected[:, :, ::2]
    features = downsampled.permute(0, 2, 1).contiguous()

    out_path = os.path.join(PROCESSED_DIR, f'lista_logpow_{tid}.pt')
    torch.save(features.to(torch.float16), out_path)

    labels = np.stack([rasterize_lidar_to_polar_grid(lidar[i]) for i in range(N)])
    label_path = os.path.join(PROCESSED_DIR, f'lista_label_{tid}.pt')
    torch.save(torch.from_numpy(labels), label_path)

    print(f'    Features: {features.shape}, Labels: {labels.shape}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bf = FFTBeamformer(N_az=N_AZ_LISTA).to(device)
    bf.eval()
    print(f'Processing {len(ALL_TRAJS)} trajectories on {device}')
    for tid in ALL_TRAJS:
        process_trajectory(tid, bf, device)
    print('Done.')


if __name__ == '__main__':
    main()
