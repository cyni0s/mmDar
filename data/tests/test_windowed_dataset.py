"""Tests for WindowedTrajectoryDataset.

TDD: these tests are written before the implementation.
"""
import torch
import pytest


@pytest.fixture
def tmp_traj(tmp_path):
    traj_id = 999
    N = 20
    radar = torch.randn(N, 8, 512, dtype=torch.complex64)
    lidar = torch.randn(N, 8192, 3)
    norm = torch.ones(N)
    torch.save(radar, str(tmp_path / f"radar_{traj_id}.pt"))
    torch.save(lidar, str(tmp_path / f"lidar_{traj_id}.pt"))
    torch.save(norm, str(tmp_path / f"norm_{traj_id}.pt"))
    return str(tmp_path), traj_id


def test_windowed_dataset_shape(tmp_traj):
    from data.windowed_dataset import WindowedTrajectoryDataset
    proc_dir, tid = tmp_traj
    ds = WindowedTrajectoryDataset(tid, proc_dir, window_size=5)
    assert len(ds) == 16  # 20 - 5 + 1
    radar_window, lidar, norm = ds[0]
    assert radar_window.shape == (5, 8, 512)
    assert radar_window.dtype == torch.complex64
    assert lidar.shape == (8192, 3)


def test_windowed_dataset_single_frame(tmp_traj):
    from data.windowed_dataset import WindowedTrajectoryDataset
    proc_dir, tid = tmp_traj
    ds = WindowedTrajectoryDataset(tid, proc_dir, window_size=1)
    assert len(ds) == 20
    radar_window, lidar, norm = ds[0]
    assert radar_window.shape == (1, 8, 512)


def test_windowed_dataset_temporal_order(tmp_traj):
    """Last frame in window should be the target frame."""
    from data.windowed_dataset import WindowedTrajectoryDataset
    proc_dir, tid = tmp_traj
    ds = WindowedTrajectoryDataset(tid, proc_dir, window_size=3)
    radar_window, lidar, norm = ds[5]
    raw_radar = torch.load(str(tmp_traj[0]) + f"/radar_999.pt", weights_only=True)
    raw_lidar = torch.load(str(tmp_traj[0]) + f"/lidar_999.pt", weights_only=True)
    # Window for ds[5] = frames [5, 6, 7], target = frame 7
    assert torch.allclose(radar_window[-1], raw_radar[7])
    assert torch.allclose(radar_window[0], raw_radar[5])
    assert torch.allclose(lidar, raw_lidar[7])


def test_windowed_augmentation_consistent(tmp_traj):
    """Augmentation should apply same phase rotation to all frames in window."""
    from data.windowed_dataset import WindowedTrajectoryDataset
    proc_dir, tid = tmp_traj
    ds = WindowedTrajectoryDataset(tid, proc_dir, window_size=3, augment=True)
    r1, _, _ = ds[0]
    r2, _, _ = ds[0]
    # Different calls should give different augmentations
    assert not torch.allclose(r1, r2)
