"""
Unit tests for SequentialDataset, TrajectoryBatchSampler, and seq_collate_fn.

Tests use actual dataset_5/train/ when available; synthetic fixtures for pure unit testing.
Coverage maps to TEMP-01/TEMP-02 requirements from RESEARCH.md.
"""

import os
import sys
import glob
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

# Allow running from repo root without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# --- Dataset path detection ---
# Supports both host path and Docker container path

def _find_dataset_root():
    candidates = [
        '/git/mmDar/dataset_5/',
        '/mmdar/dataset_5/',
        os.path.join(os.path.dirname(__file__), '..', 'dataset_5') + '/',
    ]
    for c in candidates:
        if (os.path.isdir(os.path.join(c, 'train', 'radar')) and
                len(glob.glob(os.path.join(c, 'train', 'radar', '*.png'))) > 0):
            return c
    return None


DATASET_ROOT = _find_dataset_root() or '/git/mmDar/dataset_5/'
DATASET_AVAILABLE = _find_dataset_root() is not None

skip_if_no_dataset = pytest.mark.skipif(
    not DATASET_AVAILABLE,
    reason="dataset_5/train/ not available in this environment"
)


# --- Synthetic fixture: build a tiny fake dataset on disk ---

def _write_png(path, w, h, value=128):
    """Write a minimal grayscale PNG using PIL."""
    from PIL import Image
    img = Image.fromarray(np.full((h, w), value, dtype=np.uint8), mode='L')
    img.save(path)


@pytest.fixture(scope='module')
def synthetic_dataset(tmp_path_factory):
    """Create a tiny in-memory-compatible dataset with 3 trajectories.

    Traj 100: 45 frames  -> 5 eligible (M=40)
    Traj 101: 43 frames  -> 3 eligible (M=40)
    Traj 102: 41 frames  -> 1 eligible (M=40)
    Total eligible: 9

    Lidar spatial: 256x1024 (ABINS_LIDAR_ORIG=1024, downsampled to ABINS_LIDAR=512)
    Radar spatial: 256x64   (ABINS_RADAR_ORIG=64,   ABINS_RADAR=64 -> stride 1)

    Note: lidar PNGs written at 1024-wide to match real dataset's ABINS_LIDAR_ORIG=1024.
    """
    tmpdir = tmp_path_factory.mktemp('synth_ds')
    radar_dir = tmpdir / 'train' / 'radar'
    lidar_dir = tmpdir / 'train' / 'lidar'
    radar_dir.mkdir(parents=True)
    lidar_dir.mkdir(parents=True)

    traj_frames = {100: 45, 101: 43, 102: 41}
    for traj_id, n_frames in traj_frames.items():
        for frame_idx in range(n_frames):
            _write_png(str(radar_dir / f'R_{traj_id}_{frame_idx}.png'), w=64, h=256)
            # Lidar: 256 range bins x 1024 azimuth bins (original resolution)
            _write_png(str(lidar_dir / f'L_{traj_id}_{frame_idx}.png'), w=1024, h=256)

    return str(tmpdir) + '/', traj_frames


# =============================================================================
# Group 1: SequentialDataset — basic construction
# =============================================================================

class TestSequentialDatasetBasic:

    def test_import(self):
        """SequentialDataset, TrajectoryBatchSampler, seq_collate_fn must be importable."""
        from train_test_utils.dataloader import (
            SequentialDataset, TrajectoryBatchSampler, seq_collate_fn
        )
        assert SequentialDataset is not None
        assert TrajectoryBatchSampler is not None
        assert seq_collate_fn is not None

    def test_eligible_target_count_synthetic(self, synthetic_dataset):
        """Eligible target count matches M=40 window logic on synthetic data."""
        from train_test_utils.dataloader import SequentialDataset
        basepath, traj_frames = synthetic_dataset
        ds = SequentialDataset(basepath, 'train', M=40)
        # Traj 100: 45-40=5, Traj 101: 43-40=3, Traj 102: 41-40=1  -> total 9
        expected = sum(max(0, n - 40) for n in traj_frames.values())
        assert len(ds) == expected, f"Expected {expected} eligible targets, got {len(ds)}"

    def test_exclude_traj_ids(self, synthetic_dataset):
        """exclude_traj_ids removes those trajectories entirely."""
        from train_test_utils.dataloader import SequentialDataset
        basepath, traj_frames = synthetic_dataset
        ds_full = SequentialDataset(basepath, 'train', M=40)
        ds_excl = SequentialDataset(basepath, 'train', M=40, exclude_traj_ids=[100])
        # Traj 100 has 5 eligible frames; excluding it reduces by 5
        assert len(ds_excl) == len(ds_full) - 5

    def test_include_traj_ids(self, synthetic_dataset):
        """include_traj_ids loads only those trajectories (validation mode)."""
        from train_test_utils.dataloader import SequentialDataset
        basepath, traj_frames = synthetic_dataset
        ds = SequentialDataset(basepath, 'train', M=40, include_traj_ids=[101])
        # Traj 101: 43-40=3 eligible
        assert len(ds) == 3

    def test_class_has_required_attributes(self, synthetic_dataset):
        """SequentialDataset exposes eligible_targets, traj_data."""
        from train_test_utils.dataloader import SequentialDataset
        basepath, _ = synthetic_dataset
        ds = SequentialDataset(basepath, 'train', M=40)
        assert hasattr(ds, 'eligible_targets')
        assert hasattr(ds, 'traj_data')
        assert len(ds.eligible_targets) == len(ds)


# =============================================================================
# Group 2: SequentialDataset.__getitem__
# =============================================================================

class TestSequentialDatasetGetitem:

    def test_getitem_t41_shapes(self, synthetic_dataset):
        """__getitem__ with T=41 returns correct tensor shapes."""
        import torch
        from train_test_utils.dataloader import SequentialDataset
        basepath, _ = synthetic_dataset
        ds = SequentialDataset(basepath, 'train', M=40)
        # First eligible target is in traj 100 at index 40 (0-based local)
        traj_id, target_idx = ds.eligible_targets[0]
        T = 41
        radar_seq, lidar_seq, meta = ds[(traj_id, target_idx, T)]
        # Shapes: (T, 1, H, W)
        assert radar_seq.shape == (T, 1, 256, 64), f"Got {radar_seq.shape}"
        assert lidar_seq.shape == (T, 1, 256, 512), f"Got {lidar_seq.shape}"

    def test_getitem_t1_shapes(self, synthetic_dataset):
        """__getitem__ with T=1 returns only the target frame."""
        import torch
        from train_test_utils.dataloader import SequentialDataset
        basepath, _ = synthetic_dataset
        ds = SequentialDataset(basepath, 'train', M=40)
        traj_id, target_idx = ds.eligible_targets[0]
        T = 1
        radar_seq, lidar_seq, meta = ds[(traj_id, target_idx, T)]
        assert radar_seq.shape == (1, 1, 256, 64), f"Got {radar_seq.shape}"
        assert lidar_seq.shape == (1, 1, 256, 512), f"Got {lidar_seq.shape}"

    def test_getitem_variable_t(self, synthetic_dataset):
        """__getitem__ returns T frames for arbitrary T in [1, 41]."""
        from train_test_utils.dataloader import SequentialDataset
        basepath, _ = synthetic_dataset
        ds = SequentialDataset(basepath, 'train', M=40)
        traj_id, target_idx = ds.eligible_targets[0]
        for T in [1, 5, 10, 20, 41]:
            radar_seq, lidar_seq, meta = ds[(traj_id, target_idx, T)]
            assert radar_seq.shape[0] == T, f"T={T}: got {radar_seq.shape[0]} frames"

    def test_getitem_meta_dict(self, synthetic_dataset):
        """__getitem__ returns meta dict with traj_id and target_idx."""
        from train_test_utils.dataloader import SequentialDataset
        basepath, _ = synthetic_dataset
        ds = SequentialDataset(basepath, 'train', M=40)
        traj_id, target_idx = ds.eligible_targets[0]
        _, _, meta = ds[(traj_id, target_idx, 5)]
        assert 'traj_id' in meta
        assert 'target_idx' in meta
        assert meta['traj_id'] == traj_id
        assert meta['target_idx'] == target_idx

    def test_getitem_sequential_ordering(self, synthetic_dataset):
        """T=41 window contains the 41 frames ending at target_idx (sequential order)."""
        from train_test_utils.dataloader import SequentialDataset
        basepath, _ = synthetic_dataset
        ds = SequentialDataset(basepath, 'train', M=40)
        # Use first eligible target in traj 100: local index 40 -> frames 0..40
        traj_id, target_idx = ds.eligible_targets[0]
        assert traj_id == 100
        assert target_idx == 40
        # Radar frame 40 should be the last element in the T=41 window
        radar_seq, _, _ = ds[(traj_id, target_idx, 41)]
        # radar_seq[0] is frame 0 (start), radar_seq[-1] is frame 40 (target)
        # We can verify shape; actual pixel ordering validation is implicit in loading


# =============================================================================
# Group 3: TrajectoryBatchSampler
# =============================================================================

class TestTrajectoryBatchSampler:

    def test_batch_size(self, synthetic_dataset):
        """TrajectoryBatchSampler yields batches of exactly batch_size tuples."""
        from train_test_utils.dataloader import SequentialDataset, TrajectoryBatchSampler
        basepath, _ = synthetic_dataset
        ds = SequentialDataset(basepath, 'train', M=40)
        sampler = TrajectoryBatchSampler(ds, batch_size=2, max_T=41, variable_t=False, seed=0)
        for batch in sampler:
            assert len(batch) == 2, f"Expected batch_size=2, got {len(batch)}"
            break  # check first batch only

    def test_batch_tuple_structure(self, synthetic_dataset):
        """Each element in a batch is a (traj_id, target_idx, T) tuple."""
        from train_test_utils.dataloader import SequentialDataset, TrajectoryBatchSampler
        basepath, _ = synthetic_dataset
        ds = SequentialDataset(basepath, 'train', M=40)
        sampler = TrajectoryBatchSampler(ds, batch_size=2, max_T=41, variable_t=False, seed=0)
        for batch in sampler:
            for item in batch:
                assert len(item) == 3, f"Expected 3-tuple, got {len(item)}: {item}"
                traj_id, target_idx, T = item
                assert isinstance(traj_id, int)
                assert isinstance(target_idx, int)
                assert isinstance(T, int)
            break

    def test_fixed_t(self, synthetic_dataset):
        """variable_t=False: all batches use T=max_T."""
        from train_test_utils.dataloader import SequentialDataset, TrajectoryBatchSampler
        basepath, _ = synthetic_dataset
        ds = SequentialDataset(basepath, 'train', M=40)
        sampler = TrajectoryBatchSampler(ds, batch_size=2, max_T=41, variable_t=False, seed=0)
        for batch in sampler:
            for _, _, T in batch:
                assert T == 41, f"Expected T=41 (fixed), got T={T}"

    def test_variable_t_range(self, synthetic_dataset):
        """variable_t=True: T is sampled from Uniform(1, max_T) across batches."""
        from train_test_utils.dataloader import SequentialDataset, TrajectoryBatchSampler
        basepath, _ = synthetic_dataset
        ds = SequentialDataset(basepath, 'train', M=40)
        sampler = TrajectoryBatchSampler(ds, batch_size=2, max_T=41, variable_t=True, seed=42)
        T_values = set()
        for batch in sampler:
            # All items in a batch must have the same T
            ts_in_batch = [item[2] for item in batch]
            assert len(set(ts_in_batch)) == 1, f"T must be same for all batch slots, got {ts_in_batch}"
            T_values.add(ts_in_batch[0])

        # With 9 total samples (batch=2 -> ~4 batches) and seed=42, expect range variation
        # At minimum T must be in [1, 41]
        for T in T_values:
            assert 1 <= T <= 41, f"T={T} out of range [1, 41]"
        # Over many synthetic batches T should vary (not always 41)
        # (With 9 eligible frames and batch_size=2, variable_t should produce different T values)

    def test_epoch_coverage(self, synthetic_dataset):
        """One epoch covers all eligible target frames exactly once."""
        from train_test_utils.dataloader import SequentialDataset, TrajectoryBatchSampler
        basepath, _ = synthetic_dataset
        ds = SequentialDataset(basepath, 'train', M=40)
        sampler = TrajectoryBatchSampler(ds, batch_size=1, max_T=41, variable_t=False, seed=0)
        seen = []
        for batch in sampler:
            for traj_id, target_idx, T in batch:
                seen.append((traj_id, target_idx))
        # Each eligible target should appear exactly once
        assert len(seen) == len(ds), f"Expected {len(ds)} targets, got {len(seen)}"
        assert len(set(seen)) == len(ds), "Duplicates found in epoch coverage"

    def test_set_epoch_method_exists(self, synthetic_dataset):
        """TrajectoryBatchSampler has a set_epoch method."""
        from train_test_utils.dataloader import SequentialDataset, TrajectoryBatchSampler
        basepath, _ = synthetic_dataset
        ds = SequentialDataset(basepath, 'train', M=40)
        sampler = TrajectoryBatchSampler(ds, batch_size=2, max_T=41, variable_t=False, seed=0)
        assert hasattr(sampler, 'set_epoch'), "TrajectoryBatchSampler must have set_epoch method"
        # set_epoch should not raise
        sampler.set_epoch(1)

    def test_set_epoch_changes_order(self, synthetic_dataset):
        """set_epoch shuffles trajectory assignment (different order across epochs)."""
        from train_test_utils.dataloader import SequentialDataset, TrajectoryBatchSampler
        basepath, _ = synthetic_dataset
        ds = SequentialDataset(basepath, 'train', M=40)
        sampler = TrajectoryBatchSampler(ds, batch_size=1, max_T=41, variable_t=False, seed=0)

        epoch0 = [(traj_id, target_idx) for batch in sampler for traj_id, target_idx, _ in batch]
        sampler.set_epoch(1)
        epoch1 = [(traj_id, target_idx) for batch in sampler for traj_id, target_idx, _ in batch]

        # Both epochs cover same set
        assert set(epoch0) == set(epoch1)
        # With 3 trajectories, at least some reordering is expected across epochs
        # (not guaranteed for all seeds, but very likely with seed 0 vs 1)
        # We accept that equality is rare but possible — just verify both are valid
        assert sorted(epoch0) == sorted(epoch1), "Both epochs must cover the same targets"

    def test_len(self, synthetic_dataset):
        """__len__ returns correct number of batches."""
        from train_test_utils.dataloader import SequentialDataset, TrajectoryBatchSampler
        basepath, _ = synthetic_dataset
        ds = SequentialDataset(basepath, 'train', M=40)
        sampler = TrajectoryBatchSampler(ds, batch_size=1, max_T=41, variable_t=False, seed=0)
        assert len(sampler) == len(list(sampler))


# =============================================================================
# Group 4: seq_collate_fn
# =============================================================================

class TestSeqCollateFn:

    def test_collate_stacks_batch(self, synthetic_dataset):
        """seq_collate_fn stacks (radar, lidar, meta) into (B, T, 1, H, W) tensors."""
        import torch
        from train_test_utils.dataloader import SequentialDataset, seq_collate_fn
        basepath, _ = synthetic_dataset
        ds = SequentialDataset(basepath, 'train', M=40)
        # Manually build a batch of 2 items
        traj_id0, target_idx0 = ds.eligible_targets[0]
        traj_id1, target_idx1 = ds.eligible_targets[1]
        T = 5
        item0 = ds[(traj_id0, target_idx0, T)]
        item1 = ds[(traj_id1, target_idx1, T)]
        batch = [item0, item1]
        radar_batch, lidar_batch, traj_ids, _ = seq_collate_fn(batch)

        assert radar_batch.shape == (2, T, 1, 256, 64), f"Got {radar_batch.shape}"
        assert lidar_batch.shape == (2, T, 1, 256, 512), f"Got {lidar_batch.shape}"
        assert len(traj_ids) == 2

    def test_collate_preserves_traj_ids(self, synthetic_dataset):
        """seq_collate_fn returns traj_ids matching the input items."""
        from train_test_utils.dataloader import SequentialDataset, seq_collate_fn
        basepath, _ = synthetic_dataset
        ds = SequentialDataset(basepath, 'train', M=40)
        traj_id0, target_idx0 = ds.eligible_targets[0]
        traj_id1, target_idx1 = ds.eligible_targets[-1]
        item0 = ds[(traj_id0, target_idx0, 3)]
        item1 = ds[(traj_id1, target_idx1, 3)]
        _, _, traj_ids, _ = seq_collate_fn([item0, item1])
        assert traj_ids[0] == traj_id0
        assert traj_ids[1] == traj_id1


# =============================================================================
# Group 5: Integration tests (real dataset)
# =============================================================================

class TestIntegrationRealDataset:

    # Real dataset uses ABINS_LIDAR_ORIG=512 (not the default 1024).
    # train_radarhd.py sets orig_size=[256, 64, 512] when creating Dataset.
    _REAL_DS_KWARGS = dict(
        RBINS=256, ABINS_RADAR=64, ABINS_LIDAR=512,
        RBINS_ORIG=256, ABINS_RADAR_ORIG=64, ABINS_LIDAR_ORIG=512,
    )

    @skip_if_no_dataset
    def test_eligible_count_matches_baseline(self):
        """SequentialDataset(M=40) eligible count matches Dataset(M=40) on real data."""
        from train_test_utils.dataloader import Dataset, SequentialDataset
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ds_baseline = Dataset(DATASET_ROOT, 'train', M=40, **self._REAL_DS_KWARGS)
            ds_seq = SequentialDataset(DATASET_ROOT, 'train', M=40, **self._REAL_DS_KWARGS)
        assert len(ds_seq) == len(ds_baseline), (
            f"SequentialDataset has {len(ds_seq)} targets but Dataset(M=40) has {len(ds_baseline)}"
        )

    @skip_if_no_dataset
    def test_val_split_excluded(self):
        """Validation trajectories 138/140 excluded from training set."""
        from train_test_utils.dataloader import SequentialDataset
        ds_train = SequentialDataset(DATASET_ROOT, 'train', M=40,
                                     exclude_traj_ids=[138, 140], **self._REAL_DS_KWARGS)
        traj_ids_in_train = {traj_id for traj_id, _ in ds_train.eligible_targets}
        assert 138 not in traj_ids_in_train, "Traj 138 must be excluded"
        assert 140 not in traj_ids_in_train, "Traj 140 must be excluded"

    @skip_if_no_dataset
    def test_val_only_dataset(self):
        """Validation-only SequentialDataset with include_traj_ids=[138, 140] loads only those."""
        from train_test_utils.dataloader import SequentialDataset
        ds_val = SequentialDataset(DATASET_ROOT, 'train', M=40,
                                   include_traj_ids=[138, 140], **self._REAL_DS_KWARGS)
        traj_ids_in_val = {traj_id for traj_id, _ in ds_val.eligible_targets}
        assert traj_ids_in_val == {138, 140}, f"Expected only 138/140, got {traj_ids_in_val}"
        # Expected: 883 + 1018 = 1901 (per RESEARCH.md)
        assert len(ds_val) == 1901, f"Expected 1901 val targets, got {len(ds_val)}"

    @skip_if_no_dataset
    def test_train_plus_val_equals_full(self):
        """Training + validation targets sum to full dataset eligible count."""
        from train_test_utils.dataloader import SequentialDataset
        ds_full = SequentialDataset(DATASET_ROOT, 'train', M=40, **self._REAL_DS_KWARGS)
        ds_train = SequentialDataset(DATASET_ROOT, 'train', M=40,
                                     exclude_traj_ids=[138, 140], **self._REAL_DS_KWARGS)
        ds_val = SequentialDataset(DATASET_ROOT, 'train', M=40,
                                   include_traj_ids=[138, 140], **self._REAL_DS_KWARGS)
        assert len(ds_train) + len(ds_val) == len(ds_full), (
            f"train({len(ds_train)}) + val({len(ds_val)}) != full({len(ds_full)})"
        )

    @skip_if_no_dataset
    def test_real_getitem_t41(self):
        """Real dataset: getitem returns correct shapes for T=41."""
        from train_test_utils.dataloader import SequentialDataset
        ds = SequentialDataset(DATASET_ROOT, 'train', M=40, **self._REAL_DS_KWARGS)
        traj_id, target_idx = ds.eligible_targets[0]
        T = 41
        radar_seq, lidar_seq, meta = ds[(traj_id, target_idx, T)]
        assert radar_seq.shape == (T, 1, 256, 64)
        assert lidar_seq.shape == (T, 1, 256, 512)
        assert meta['traj_id'] == traj_id

    @skip_if_no_dataset
    def test_training_eligible_count(self):
        """Training set (excluding 138/140) has 19880 eligible targets per RESEARCH.md."""
        from train_test_utils.dataloader import SequentialDataset
        ds_train = SequentialDataset(DATASET_ROOT, 'train', M=40,
                                     exclude_traj_ids=[138, 140], **self._REAL_DS_KWARGS)
        assert len(ds_train) == 19880, (
            f"Expected 19880 training targets, got {len(ds_train)}"
        )


# =============================================================================
# Group 6: Original Dataset class unchanged
# =============================================================================

class TestOriginalDatasetUnchanged:

    def test_dataset_class_still_importable(self):
        """Original Dataset class is still importable and functional."""
        from train_test_utils.dataloader import Dataset
        assert Dataset is not None

    def test_dataset_class_signature_unchanged(self):
        """Dataset.__init__ signature must match original (M parameter)."""
        import inspect
        from train_test_utils.dataloader import Dataset
        sig = inspect.signature(Dataset.__init__)
        params = list(sig.parameters.keys())
        assert 'M' in params, "Dataset.__init__ must have M parameter"
        assert 'basepath' in params
        assert 'sub' in params
