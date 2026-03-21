"""
Tests for v2/data/split.py and v2/data/dataset.py.

Split tests do not require .pt files.
Dataset tests use synthetic in-memory .pt files via the mock_processed_dir fixture.
No real data, Docker, or network access needed.
"""

import os

import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_processed_dir(tmp_path):
    """
    Create a minimal fake processed/ directory with .pt files for trajectory 999.

    radar_999.pt  : (10, 8, 512) complex64  — random complex IQ data
    lidar_999.pt  : (10, 8192, 3) float32   — random point cloud
    norm_999.pt   : (10,) float32           — random normalization factors

    Returns tmp_path as string (processed_dir argument for TrajectoryDataset).
    """
    n_frames = 10
    real = torch.randn(n_frames, 8, 512)
    imag = torch.randn(n_frames, 8, 512)
    radar = torch.complex(real, imag)                         # (10, 8, 512) complex64
    lidar = torch.randn(n_frames, 8192, 3)                    # (10, 8192, 3) float32
    norm = torch.rand(n_frames)                               # (10,) float32

    torch.save(radar, tmp_path / "radar_999.pt")
    torch.save(lidar, tmp_path / "lidar_999.pt")
    torch.save(norm, tmp_path / "norm_999.pt")

    return str(tmp_path)


# ---------------------------------------------------------------------------
# Split tests (no .pt files needed)
# ---------------------------------------------------------------------------

def test_split_no_overlap():
    """No trajectory appears in more than one split."""
    from v2.data.split import TRAIN_TRAJS, VAL_TRAJS, TEST_TRAJS
    assert set(TRAIN_TRAJS).isdisjoint(set(VAL_TRAJS)), \
        "TRAIN and VAL share trajectories"
    assert set(TRAIN_TRAJS).isdisjoint(set(TEST_TRAJS)), \
        "TRAIN and TEST share trajectories"
    assert set(VAL_TRAJS).isdisjoint(set(TEST_TRAJS)), \
        "VAL and TEST share trajectories"


def test_split_complete():
    """Union of all splits equals exactly 44 trajectory IDs."""
    from v2.data.split import TRAIN_TRAJS, VAL_TRAJS, TEST_TRAJS
    all_trajs = set(TRAIN_TRAJS) | set(VAL_TRAJS) | set(TEST_TRAJS)
    assert len(all_trajs) == 44, \
        f"Expected 44 unique trajectory IDs, got {len(all_trajs)}"


def test_test_set_sealed():
    """TEST_TRAJS is exactly 19 trajectories and contains required baseline IDs."""
    from v2.data.split import TEST_TRAJS
    assert len(TEST_TRAJS) == 19, \
        f"TEST_TRAJS must have exactly 19 elements, got {len(TEST_TRAJS)}"
    # Spot-check key baseline trajectories
    assert 117 in TEST_TRAJS, "Trajectory 117 missing from TEST_TRAJS"
    assert 124 in TEST_TRAJS, "Trajectory 124 missing from TEST_TRAJS"
    assert 227 in TEST_TRAJS, "Trajectory 227 missing from TEST_TRAJS"


def test_validate_split_passes():
    """validate_split() should not raise."""
    from v2.data.split import validate_split
    validate_split()  # raises if any assertion fails


def test_get_split_returns_correct_lists():
    """get_split() returns the right constant for each split name."""
    from v2.data.split import get_split, TRAIN_TRAJS, VAL_TRAJS, TEST_TRAJS
    assert get_split("train") is TRAIN_TRAJS
    assert get_split("val") is VAL_TRAJS
    assert get_split("test") is TEST_TRAJS


def test_get_split_raises_on_unknown():
    """get_split() raises ValueError for unknown split names."""
    from v2.data.split import get_split
    with pytest.raises(ValueError, match="Unknown split"):
        get_split("bogus")


# ---------------------------------------------------------------------------
# Dataset length and output shape tests
# ---------------------------------------------------------------------------

def test_dataset_len(mock_processed_dir):
    """TrajectoryDataset.__len__ matches the number of frames in the radar tensor."""
    from v2.data.dataset import TrajectoryDataset
    ds = TrajectoryDataset(999, mock_processed_dir, augment=False)
    assert len(ds) == 10


def test_dataset_output_shape(mock_processed_dir):
    """__getitem__ returns correct shapes and dtypes."""
    from v2.data.dataset import TrajectoryDataset
    ds = TrajectoryDataset(999, mock_processed_dir, augment=False)
    r, l, nf = ds[0]
    # Radar: (8, 512) complex64
    assert r.shape == (8, 512), f"radar shape {r.shape} != (8, 512)"
    assert r.dtype == torch.complex64, f"radar dtype {r.dtype} != complex64"
    # Lidar: (8192, 3) float32
    assert l.shape == (8192, 3), f"lidar shape {l.shape} != (8192, 3)"
    assert l.dtype == torch.float32, f"lidar dtype {l.dtype} != float32"
    # Norm factor: scalar float32
    assert nf.dtype == torch.float32, f"norm_factor dtype {nf.dtype} != float32"
    assert nf.shape == (), f"norm_factor should be scalar, got shape {nf.shape}"


# ---------------------------------------------------------------------------
# Augmentation tests
# ---------------------------------------------------------------------------

def test_augmentation_changes_radar(mock_processed_dir):
    """With augment=True, two calls on same index produce different radar tensors."""
    from v2.data.dataset import TrajectoryDataset
    ds = TrajectoryDataset(999, mock_processed_dir, augment=True)
    r1, _, _ = ds[0]
    r2, _, _ = ds[0]
    # Random augmentation must produce different outputs with overwhelming probability
    assert not torch.allclose(r1, r2), \
        "Augmented outputs should differ across calls (random phase/noise/shift)"


def test_no_augment_deterministic(mock_processed_dir):
    """With augment=False, same index returns identical radar tensors each call."""
    from v2.data.dataset import TrajectoryDataset
    ds = TrajectoryDataset(999, mock_processed_dir, augment=False)
    r1, _, _ = ds[0]
    r2, _, _ = ds[0]
    assert torch.allclose(r1, r2), \
        "Without augmentation, same index must return identical tensors"


def test_augmentation_preserves_shape_dtype(mock_processed_dir):
    """Augmented radar output keeps (8, 512) complex64 shape and dtype."""
    from v2.data.dataset import TrajectoryDataset
    ds = TrajectoryDataset(999, mock_processed_dir, augment=True)
    r, l, _ = ds[0]
    assert r.shape == (8, 512), f"Augmented radar shape {r.shape} != (8, 512)"
    assert r.dtype == torch.complex64, f"Augmented radar dtype {r.dtype} != complex64"
    assert l.shape == (8192, 3)
    assert l.dtype == torch.float32


def test_augmentation_lidar_unchanged(mock_processed_dir):
    """Augmentation is only applied to radar; lidar is not modified."""
    from v2.data.dataset import TrajectoryDataset
    ds_aug = TrajectoryDataset(999, mock_processed_dir, augment=True)
    ds_no = TrajectoryDataset(999, mock_processed_dir, augment=False)
    _, l_aug, _ = ds_aug[3]
    _, l_no, _ = ds_no[3]
    assert torch.allclose(l_aug, l_no), "Lidar must not change with augmentation"
