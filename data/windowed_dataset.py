"""Windowed trajectory dataset for temporal fusion training.

Returns N consecutive radar frames + lidar label for the last (target) frame.

Window convention:
  ds[i] -> frames [i, i+1, ..., i+window_size-1]
  target frame = frame i+window_size-1 (last in window)
  lidar and norm are taken from the target frame.
"""
import os
import random
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from data.split import TRAIN_TRAJS as _DEFAULT_TRAIN, VAL_TRAJS as _DEFAULT_VAL, TEST_TRAJS as _DEFAULT_TEST


class WindowedTrajectoryDataset(Dataset):
    """Returns (window_size, 8, 512) radar windows + last-frame lidar.

    Args:
        traj_id: trajectory ID (int)
        processed_dir: directory containing radar_{tid}.pt, lidar_{tid}.pt, norm_{tid}.pt
        window_size: number of consecutive frames to include (default 5)
        augment: if True, apply temporally-consistent IQ augmentation (phase rotation +
                 per-frame independent Gaussian noise). NO range shift.
        noise_sigma: standard deviation of additive complex Gaussian noise (default 0.01)

    Returns (per __getitem__):
        radar_window: (window_size, 8, 512) complex64
        lidar:        (8192, 3) float32  -- from the target (last) frame
        norm:         scalar float32     -- from the target (last) frame
    """

    def __init__(
        self,
        traj_id: int,
        processed_dir: str,
        window_size: int = 5,
        augment: bool = False,
        noise_sigma: float = 0.01,
    ):
        super().__init__()
        assert window_size >= 1, f"window_size must be >= 1, got {window_size}"
        self.window_size = window_size
        self.augment = augment
        self.noise_sigma = noise_sigma

        self.radar = torch.load(
            os.path.join(processed_dir, f"radar_{traj_id}.pt"), weights_only=True
        )
        self.lidar = torch.load(
            os.path.join(processed_dir, f"lidar_{traj_id}.pt"), weights_only=True
        )
        self.norm_factors = torch.load(
            os.path.join(processed_dir, f"norm_{traj_id}.pt"), weights_only=True
        )

        assert self.radar.shape[0] == self.lidar.shape[0] == self.norm_factors.shape[0], (
            f"Frame count mismatch for traj {traj_id}: "
            f"radar={self.radar.shape[0]}, lidar={self.lidar.shape[0]}, "
            f"norm={self.norm_factors.shape[0]}"
        )
        self.n_frames = self.radar.shape[0]

    def __len__(self) -> int:
        return max(0, self.n_frames - self.window_size + 1)

    def __getitem__(self, idx: int):
        target_idx = idx + self.window_size - 1
        radar_window = self.radar[idx : idx + self.window_size].clone()  # (W, 8, 512)
        lidar = self.lidar[target_idx]
        norm = self.norm_factors[target_idx]

        if self.augment:
            # Same phase rotation applied to all frames — temporally consistent
            theta = random.uniform(0.0, 2.0 * 3.141592653589793)
            phase_rot = torch.exp(torch.tensor(1j * theta, dtype=torch.complex64))
            radar_window = radar_window * phase_rot

            # Independent additive complex Gaussian noise per frame (models thermal noise)
            noise = self.noise_sigma * (
                torch.randn_like(radar_window.real) + 1j * torch.randn_like(radar_window.imag)
            ).to(torch.complex64)
            radar_window = radar_window + noise

        return radar_window, lidar, norm


def build_windowed_dataloaders(
    processed_dir: str,
    window_size: int = 5,
    batch_size: int = 12,
    num_workers: int = 4,
    train_trajs: list[int] | None = None,
    val_trajs: list[int] | None = None,
    test_trajs: list[int] | None = None,
) -> dict:
    """Build train/val/test DataLoaders with windowed trajectory datasets.

    Args:
        processed_dir: directory with per-trajectory .pt files
        window_size: number of consecutive radar frames per sample
        batch_size: DataLoader batch size
        num_workers: DataLoader worker processes
        train_trajs: trajectory IDs for train split (default: split.py TRAIN_TRAJS)
        val_trajs: trajectory IDs for val split (default: split.py VAL_TRAJS)
        test_trajs: trajectory IDs for test split (default: split.py TEST_TRAJS)

    Returns:
        dict with "train", "val", "test" keys, each a DataLoader.
        Each batch yields (radar_window, lidar, norm):
          - radar_window: (B, W, 8, 512) complex64
          - lidar:        (B, 8192, 3)   float32
          - norm:         (B,)           float32
    """
    split_configs = {
        "train": (train_trajs or _DEFAULT_TRAIN, True, True),
        "val":   (val_trajs or _DEFAULT_VAL,     False, False),
        "test":  (test_trajs or _DEFAULT_TEST,   False, False),
    }
    loaders = {}
    for split_name, (traj_ids, augment, shuffle) in split_configs.items():
        datasets = []
        for tid in traj_ids:
            radar_path = os.path.join(processed_dir, f"radar_{tid}.pt")
            if not os.path.isfile(radar_path):
                print(
                    f"[build_windowed_dataloaders] WARNING: {radar_path} not found, "
                    f"skipping traj {tid}"
                )
                continue
            datasets.append(
                WindowedTrajectoryDataset(
                    tid, processed_dir, window_size=window_size, augment=augment
                )
            )
        if not datasets:
            raise RuntimeError(
                f"No trajectories found for '{split_name}' split in {processed_dir}"
            )
        loaders[split_name] = DataLoader(
            ConcatDataset(datasets),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    return loaders
