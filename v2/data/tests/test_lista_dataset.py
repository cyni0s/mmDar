import torch
import pytest


def test_lista_dataset_shapes():
    """Verify dataset returns correct shapes for 41-frame stacking."""
    from v2.data.lista_dataset import LISTAOccDataset
    ds = LISTAOccDataset(traj_id=117, processed_dir='v2/data/processed', history=40)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (41, 256, 512), f"Expected (41, 256, 512), got {x.shape}"
    assert y.shape == (1, 256, 512), f"Expected (1, 256, 512), got {y.shape}"
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert y.min() >= 0 and y.max() <= 1


def test_lista_dataset_stacking_order():
    """History frames should be oldest-first, current frame last."""
    from v2.data.lista_dataset import LISTAOccDataset
    ds = LISTAOccDataset(traj_id=117, processed_dir='v2/data/processed', history=40)
    x, y = ds[0]
    assert x.shape[0] == 41
