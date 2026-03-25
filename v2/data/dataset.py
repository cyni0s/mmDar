"""
v2 Dataset and DataLoader
=========================

Provides PyTorch Dataset/DataLoader wrappers for the preprocessed .pt tensor files
produced by v2/data/preprocess.py.

Expected .pt file format (per trajectory, saved by preprocess.py):
  - v2/data/processed/radar_{traj_id}.pt   shape: (N_frames, 8, 512) complex64
      Normalized complex IQ data: 8 virtual antennas × 512 range bins.
      Each frame is divided by its per-frame max magnitude before saving.
  - v2/data/processed/lidar_{traj_id}.pt   shape: (N_frames, 8192, 3) float32
      Aligned lidar point cloud after FPS to 8192 points, in metres (x, y, z).
  - v2/data/processed/norm_{traj_id}.pt    shape: (N_frames,) float32
      Per-frame normalization factor (max magnitude before normalization).
      Use to denormalize radar amplitudes if needed during training or evaluation.

Split definitions:
  See v2/data/split.py for TRAIN_TRAJS, VAL_TRAJS, TEST_TRAJS.
  Split is trajectory-level (no frame-level leakage between sets).

IQ-domain augmentation strategy (train split only):
  Applied in the complex domain to preserve phase information. Three transforms,
  all applied when augment=True:

  1. Global phase rotation: r = r * exp(j * theta), theta ~ Uniform(0, 2*pi)
     Physics rationale: Scene has no preferred absolute phase reference. A global
     phase rotation is equivalent to a different receiver phase reference and
     should produce the same point cloud output. This is the single most impactful
     augmentation for complex IQ data.

  2. Additive complex Gaussian noise: r = r + sigma * (N_real + j*N_imag)
     sigma = 0.01 (relative to normalized max=1 input).
     Physics rationale: Models thermal noise and quantization noise in the ADC.
     sigma=0.01 is conservative (SNR ~40 dB) to avoid corrupting weak returns.

  3. Circular range shift: r = roll(r, shift, dims=-1), shift ~ Uniform(-2, 2)
     Physics rationale: Small range offsets (<= 2 bins = 2 * ~4.2cm = 8.4cm)
     model timing jitter and small radar-vehicle motion artifacts.
     Circular roll preserves all energy and dtype.

Note: norm_factor scalar is returned as-is (no augmentation applied to it).
"""

import os
import random

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from v2.data.split import TRAIN_TRAJS, VAL_TRAJS, TEST_TRAJS, get_split


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for a single trajectory's preprocessed .pt tensors.

    Parameters
    ----------
    traj_id : int
        Trajectory ID (used to construct filenames radar_{traj_id}.pt etc.).
    processed_dir : str
        Directory containing the .pt files.
    augment : bool, optional
        If True, apply IQ-domain augmentation on each __getitem__ call.
        Default: False.
    noise_sigma : float, optional
        Standard deviation of additive complex Gaussian noise. Default: 0.01.

    Returns (per item)
    ------------------
    radar_frame : torch.Tensor, shape (8, 512), dtype torch.complex64
        Normalized complex IQ snapshot. Augmented if augment=True.
    lidar_frame : torch.Tensor, shape (8192, 3), dtype torch.float32
        Aligned lidar point cloud in metres (x, y, z).
    norm_factor : torch.Tensor, scalar, dtype torch.float32
        Per-frame normalization factor (max magnitude before normalization).
    """

    def __init__(
        self,
        traj_id: int,
        processed_dir: str,
        augment: bool = False,
        noise_sigma: float = 0.01,
    ) -> None:
        super().__init__()
        self.traj_id = traj_id
        self.augment = augment
        self.noise_sigma = noise_sigma

        radar_path = os.path.join(processed_dir, f"radar_{traj_id}.pt")
        lidar_path = os.path.join(processed_dir, f"lidar_{traj_id}.pt")
        norm_path = os.path.join(processed_dir, f"norm_{traj_id}.pt")

        self.radar = torch.load(radar_path, weights_only=True)   # (N, 8, 512) complex64
        self.lidar = torch.load(lidar_path, weights_only=True)   # (N, 8192, 3) float32
        self.norm_factors = torch.load(norm_path, weights_only=True)  # (N,) float32

        assert self.radar.shape[0] == self.lidar.shape[0] == self.norm_factors.shape[0], (
            f"Trajectory {traj_id}: frame count mismatch — "
            f"radar={self.radar.shape[0]}, lidar={self.lidar.shape[0]}, "
            f"norm={self.norm_factors.shape[0]}"
        )
        assert self.radar.dtype == torch.complex64, (
            f"radar_{traj_id}.pt must be complex64, got {self.radar.dtype}"
        )
        assert self.lidar.dtype == torch.float32, (
            f"lidar_{traj_id}.pt must be float32, got {self.lidar.dtype}"
        )

    def __len__(self) -> int:
        return self.radar.shape[0]

    def __getitem__(self, idx: int):
        radar_frame = self.radar[idx].clone()   # (8, 512) complex64
        lidar_frame = self.lidar[idx]           # (8192, 3) float32
        norm_factor = self.norm_factors[idx]    # scalar float32

        if self.augment:
            radar_frame = self._augment(radar_frame)

        return radar_frame, lidar_frame, norm_factor

    def _augment(self, r: torch.Tensor) -> torch.Tensor:
        """
        Apply IQ-domain augmentation to a single radar frame.

        Parameters
        ----------
        r : torch.Tensor, shape (8, 512), complex64

        Returns
        -------
        torch.Tensor, shape (8, 512), complex64
        """
        # 1. Global phase rotation: theta ~ Uniform(0, 2*pi)
        theta = random.uniform(0.0, 2.0 * torch.pi)
        r = r * torch.exp(torch.tensor(1j * theta, dtype=torch.complex64))

        # 2. Additive complex Gaussian noise
        noise = self.noise_sigma * (
            torch.randn_like(r.real) + 1j * torch.randn_like(r.imag)
        ).to(torch.complex64)
        r = r + noise

        # 3. Circular range shift along the range (last) dimension
        shift = random.randint(-2, 2)
        r = torch.roll(r, shift, dims=-1)

        return r


# ---------------------------------------------------------------------------
# Occupancy Dataset class
# ---------------------------------------------------------------------------

class OccupancyTrajectoryDataset(Dataset):
    """
    PyTorch Dataset for a single trajectory with polar occupancy labels.

    Extends TrajectoryDataset to also load occ_{traj_id}.pt — rasterized
    occupancy grids used as training targets for the occupancy decoder.

    Parameters
    ----------
    traj_id : int
        Trajectory ID (used to construct filenames radar_{traj_id}.pt etc.).
    processed_dir : str
        Directory containing the .pt files.
    augment : bool, optional
        If True, apply IQ-domain augmentation to radar on each __getitem__.
        Augmentation is ONLY applied to radar — lidar and occ_label are
        returned unchanged (they represent ground-truth geometry).
        Default: False.
    noise_sigma : float, optional
        Standard deviation of additive complex Gaussian noise. Default: 0.01.

    Returns (per item)
    ------------------
    radar_frame : torch.Tensor, shape (8, 512), dtype torch.complex64
        Normalized complex IQ snapshot. Augmented if augment=True.
    lidar_frame : torch.Tensor, shape (8192, 3), dtype torch.float32
        Aligned lidar point cloud in metres (x, y, z). Used for eval
        against original point cloud metrics (Chamfer, mod-Hausdorff).
    occ_label : torch.Tensor, shape (256, 512), dtype torch.float32
        Rasterized polar occupancy grid. Training target for the occupancy
        decoder. Not augmented.
    norm_factor : torch.Tensor, scalar, dtype torch.float32
        Per-frame normalization factor (max magnitude before normalization).
    """

    def __init__(
        self,
        traj_id: int,
        processed_dir: str,
        augment: bool = False,
        noise_sigma: float = 0.01,
    ) -> None:
        super().__init__()
        self.traj_id = traj_id
        self.augment = augment
        self.noise_sigma = noise_sigma

        radar_path = os.path.join(processed_dir, f"radar_{traj_id}.pt")
        lidar_path = os.path.join(processed_dir, f"lidar_{traj_id}.pt")
        norm_path = os.path.join(processed_dir, f"norm_{traj_id}.pt")
        occ_path = os.path.join(processed_dir, f"occ_{traj_id}.pt")

        self.radar = torch.load(radar_path, weights_only=True)       # (N, 8, 512) complex64
        self.lidar = torch.load(lidar_path, weights_only=True)       # (N, 8192, 3) float32
        self.norm_factors = torch.load(norm_path, weights_only=True) # (N,) float32
        self.occ = torch.load(occ_path, weights_only=True)           # (N, 256, 512) float32

        N = self.radar.shape[0]
        assert N == self.lidar.shape[0] == self.norm_factors.shape[0] == self.occ.shape[0], (
            f"Trajectory {traj_id}: frame count mismatch — "
            f"radar={N}, lidar={self.lidar.shape[0]}, "
            f"norm={self.norm_factors.shape[0]}, occ={self.occ.shape[0]}"
        )
        assert self.radar.dtype == torch.complex64, (
            f"radar_{traj_id}.pt must be complex64, got {self.radar.dtype}"
        )
        assert self.lidar.dtype == torch.float32, (
            f"lidar_{traj_id}.pt must be float32, got {self.lidar.dtype}"
        )
        assert self.occ.dtype == torch.float32, (
            f"occ_{traj_id}.pt must be float32, got {self.occ.dtype}"
        )

    def __len__(self) -> int:
        return self.radar.shape[0]

    def __getitem__(self, idx: int):
        radar_frame = self.radar[idx].clone()   # (8, 512) complex64
        lidar_frame = self.lidar[idx]           # (8192, 3) float32
        occ_label = self.occ[idx]               # (256, 512) float32
        norm_factor = self.norm_factors[idx]    # scalar float32

        if self.augment:
            radar_frame = self._augment(radar_frame)

        return radar_frame, lidar_frame, occ_label, norm_factor

    def _augment(self, r: torch.Tensor) -> torch.Tensor:
        """
        Apply IQ-domain augmentation to a single radar frame.

        Only applied to radar. Lidar and occ_label are NOT modified.

        Parameters
        ----------
        r : torch.Tensor, shape (8, 512), complex64

        Returns
        -------
        torch.Tensor, shape (8, 512), complex64
        """
        # 1. Global phase rotation: theta ~ Uniform(0, 2*pi)
        theta = random.uniform(0.0, 2.0 * torch.pi)
        r = r * torch.exp(torch.tensor(1j * theta, dtype=torch.complex64))

        # 2. Additive complex Gaussian noise
        noise = self.noise_sigma * (
            torch.randn_like(r.real) + 1j * torch.randn_like(r.imag)
        ).to(torch.complex64)
        r = r + noise

        # 3. Circular range shift REMOVED for occupancy training.
        # Range shift moves radar features along the range axis but does NOT
        # shift the occupancy label, creating a train-time input/target mismatch.
        # Phase rotation and additive noise are label-preserving; range shift is not.

        return r


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    processed_dir: str,
    batch_size: int = 12,
    num_workers: int = 4,
) -> dict:
    """
    Build train/val/test DataLoaders from preprocessed .pt files.

    Each DataLoader wraps a ConcatDataset of per-trajectory TrajectoryDatasets.
    Only trajectories whose .pt files exist in processed_dir are included
    (missing trajectories are silently skipped with a warning).

    Parameters
    ----------
    processed_dir : str
        Directory containing radar_{id}.pt, lidar_{id}.pt, norm_{id}.pt files.
    batch_size : int, optional
        Batch size for all DataLoaders. Default: 12.
    num_workers : int, optional
        Number of DataLoader worker processes. Default: 4.

    Returns
    -------
    dict with keys "train", "val", "test", each a torch.utils.data.DataLoader.
        Each batch yields (radar, lidar, norm) tuples:
          - radar:  (B, 8, 512)   complex64
          - lidar:  (B, 8192, 3)  float32
          - norm:   (B,)          float32
    """
    split_configs = {
        "train": (TRAIN_TRAJS, True, True),    # (trajs, augment, shuffle)
        "val":   (VAL_TRAJS,   False, False),
        "test":  (TEST_TRAJS,  False, False),
    }

    loaders = {}
    for split_name, (traj_ids, augment, shuffle) in split_configs.items():
        datasets = []
        for tid in traj_ids:
            radar_path = os.path.join(processed_dir, f"radar_{tid}.pt")
            if not os.path.isfile(radar_path):
                print(
                    f"[build_dataloaders] WARNING: {radar_path} not found, "
                    f"skipping trajectory {tid} from {split_name} split."
                )
                continue
            datasets.append(
                TrajectoryDataset(tid, processed_dir, augment=augment)
            )

        if not datasets:
            print(
                f"[build_dataloaders] WARNING: No .pt files found for "
                f"'{split_name}' split. DataLoader will be empty."
            )
            # Return an empty DataLoader rather than crashing
            concat = ConcatDataset([])
        else:
            concat = ConcatDataset(datasets)

        loaders[split_name] = DataLoader(
            concat,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    return loaders


def build_occupancy_dataloaders(
    processed_dir: str,
    batch_size: int = 12,
    num_workers: int = 4,
) -> dict:
    """
    Build train/val/test DataLoaders using OccupancyTrajectoryDataset.

    Same split and shuffle logic as build_dataloaders, but each batch
    yields 4-tuples (radar, lidar, occ_label, norm). Only trajectories
    with both radar_{tid}.pt AND occ_{tid}.pt present are included.

    Parameters
    ----------
    processed_dir : str
        Directory containing radar_{id}.pt, lidar_{id}.pt, norm_{id}.pt,
        and occ_{id}.pt files.
    batch_size : int, optional
        Batch size for all DataLoaders. Default: 12.
    num_workers : int, optional
        Number of DataLoader worker processes. Default: 4.

    Returns
    -------
    dict with keys "train", "val", "test", each a torch.utils.data.DataLoader.
        Each batch yields (radar, lidar, occ_label, norm) tuples:
          - radar:     (B, 8, 512)    complex64
          - lidar:     (B, 8192, 3)  float32
          - occ_label: (B, 256, 512) float32
          - norm:      (B,)           float32
    """
    split_configs = {
        "train": (TRAIN_TRAJS, True, True),    # (trajs, augment, shuffle)
        "val":   (VAL_TRAJS,   False, False),
        "test":  (TEST_TRAJS,  False, False),
    }

    loaders = {}
    for split_name, (traj_ids, augment, shuffle) in split_configs.items():
        datasets = []
        for tid in traj_ids:
            # Check ALL 4 required files before constructing the dataset
            required = {
                "radar": os.path.join(processed_dir, f"radar_{tid}.pt"),
                "lidar": os.path.join(processed_dir, f"lidar_{tid}.pt"),
                "norm": os.path.join(processed_dir, f"norm_{tid}.pt"),
                "occ": os.path.join(processed_dir, f"occ_{tid}.pt"),
            }
            missing = [k for k, v in required.items() if not os.path.isfile(v)]
            if missing:
                print(
                    f"[build_occupancy_dataloaders] WARNING: trajectory {tid} "
                    f"missing {missing}, skipping from {split_name} split."
                )
                continue
            datasets.append(
                OccupancyTrajectoryDataset(tid, processed_dir, augment=augment)
            )

        if not datasets:
            raise RuntimeError(
                f"[build_occupancy_dataloaders] No valid trajectories found for "
                f"'{split_name}' split in {processed_dir}. "
                f"Run 'python -m v2.data.rasterize' to generate occ files."
            )

        concat = ConcatDataset(datasets)
        loaders[split_name] = DataLoader(
            concat,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    return loaders
