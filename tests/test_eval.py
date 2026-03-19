"""Tests for eval/eval_pointcloud.py — temporal_consistency metric."""

import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from eval.eval_pointcloud import temporal_consistency


def _make_pred_png(dirpath, traj_id, frame_idx, value=128):
    """Create a synthetic *_pred.png with a bright rectangle for point-cloud extraction."""
    img = np.zeros((256, 512), dtype=np.uint8)
    # Add bright pixels so polar_image_to_pointcloud returns non-empty cloud
    # Vary position slightly per frame_idx so consecutive frames differ
    row_offset = (frame_idx * 5) % 50
    img[100 + row_offset:110 + row_offset, 200:220] = value
    fname = os.path.join(dirpath, f'{traj_id}_{frame_idx}_pred.png')
    Image.fromarray(img).save(fname)
    return fname


def test_temporal_consistency_basic():
    """temporal_consistency on 5 frames from a single trajectory.

    5 frames -> 4 consecutive pairs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_id = 10
        for frame_idx in range(5):
            _make_pred_png(tmpdir, traj_id, frame_idx)

        result = temporal_consistency(tmpdir)

    assert isinstance(result, dict), "Should return a dict"
    assert set(result.keys()) >= {'mean', 'median', 'std', 'count', 'scores'}
    assert result['count'] == 4, f"Expected 4 pairs from 5 frames, got {result['count']}"
    assert isinstance(result['mean'], float)
    assert isinstance(result['median'], float)
    assert isinstance(result['std'], float)
    assert len(result['scores']) == result['count']
    assert result['mean'] >= 0.0


def test_temporal_consistency_multi_traj():
    """temporal_consistency does not cross trajectory boundaries.

    Traj 1: 3 frames -> 2 pairs
    Traj 2: 2 frames -> 1 pair
    Total: 3 pairs (not 4 which would happen if boundaries were crossed)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Trajectory 1: frames 0, 1, 2
        for frame_idx in range(3):
            _make_pred_png(tmpdir, traj_id=1, frame_idx=frame_idx)
        # Trajectory 2: frames 0, 1
        for frame_idx in range(2):
            _make_pred_png(tmpdir, traj_id=2, frame_idx=frame_idx)

        result = temporal_consistency(tmpdir)

    assert result['count'] == 3, (
        f"Expected 3 pairs (2 from traj 1 + 1 from traj 2), got {result['count']}. "
        f"Boundary crossing would give 4."
    )


def test_temporal_consistency_empty_dir():
    """temporal_consistency returns zero-count dict for directory with no pred files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = temporal_consistency(tmpdir)

    assert result['count'] == 0
    assert np.isnan(result['mean'])


def test_temporal_consistency_epoch_prefixed_filenames():
    """temporal_consistency handles epoch-prefixed filenames: {epoch}_{traj}_{frame}_pred.png."""
    with tempfile.TemporaryDirectory() as tmpdir:
        epoch = 30
        traj_id = 5
        for frame_idx in range(3):
            img = np.zeros((256, 512), dtype=np.uint8)
            img[100:110, 200:220] = 200
            fname = os.path.join(tmpdir, f'{epoch}_{traj_id}_{frame_idx}_pred.png')
            Image.fromarray(img).save(fname)

        result = temporal_consistency(tmpdir)

    # 3 frames from 1 trajectory -> 2 pairs
    assert result['count'] == 2, f"Expected 2 pairs, got {result['count']}"
