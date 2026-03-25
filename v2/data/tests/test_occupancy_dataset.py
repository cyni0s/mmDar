"""Tests for OccupancyTrajectoryDataset and build_occupancy_dataloaders."""

import os
import torch
import numpy as np
import pytest
from v2.data.rasterize import rasterize_to_polar


@pytest.fixture
def tmp_processed(tmp_path):
    """Create minimal fake processed data with occupancy labels."""
    N = 5
    traj_id = 999
    radar = torch.randn(N, 8, 512, dtype=torch.complex64)
    lidar = torch.randn(N, 8192, 3)
    norm = torch.ones(N)

    occ_list = []
    for i in range(N):
        pts = lidar[i].numpy().copy()
        pts[:10, 0] = np.random.uniform(1, 10, 10)
        pts[:10, 1] = np.random.uniform(-5, 5, 10)
        pts[:10, 2] = 0
        occ_list.append(rasterize_to_polar(pts[:10]))
    occ = torch.from_numpy(np.stack(occ_list))

    torch.save(radar, str(tmp_path / f"radar_{traj_id}.pt"))
    torch.save(lidar, str(tmp_path / f"lidar_{traj_id}.pt"))
    torch.save(norm, str(tmp_path / f"norm_{traj_id}.pt"))
    torch.save(occ, str(tmp_path / f"occ_{traj_id}.pt"))

    return str(tmp_path), traj_id


def test_occupancy_dataset_loads(tmp_processed):
    from v2.data.dataset import OccupancyTrajectoryDataset
    proc_dir, tid = tmp_processed
    ds = OccupancyTrajectoryDataset(tid, proc_dir)
    assert len(ds) == 5
    radar, lidar, occ, norm = ds[0]
    assert radar.shape == (8, 512)
    assert radar.dtype == torch.complex64
    assert lidar.shape == (8192, 3)
    assert lidar.dtype == torch.float32
    assert occ.shape == (256, 512)
    assert occ.dtype == torch.float32
    assert norm.shape == ()


def test_occupancy_dataset_augment(tmp_processed):
    from v2.data.dataset import OccupancyTrajectoryDataset
    proc_dir, tid = tmp_processed
    ds = OccupancyTrajectoryDataset(tid, proc_dir, augment=True)
    r1, _, o1, _ = ds[0]
    r2, _, o2, _ = ds[0]
    assert not torch.allclose(r1, r2), "Augmentation should change radar"
    assert torch.allclose(o1, o2), "Occupancy labels should not be augmented"
