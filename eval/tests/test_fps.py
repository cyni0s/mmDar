import torch
import numpy as np
import pytest


def test_fps_returns_correct_count():
    """FPS on 100 points requesting 10 should return exactly 10."""
    from eval.fps import fps_2d
    pts = torch.randn(100, 2)
    result = fps_2d(pts, 10, seed=0)
    assert result.shape == (10, 2)


def test_fps_fewer_than_n_returns_all():
    """When K < N, fps_2d returns all K points (no padding)."""
    from eval.fps import fps_2d
    pts = torch.randn(5, 2)
    result = fps_2d(pts, 10, seed=0)
    assert result.shape == (5, 2)


def test_fps_deterministic():
    """Same seed produces same output."""
    from eval.fps import fps_2d
    pts = torch.randn(100, 2)
    r1 = fps_2d(pts, 20, seed=0)
    r2 = fps_2d(pts, 20, seed=0)
    assert torch.allclose(r1, r2)


def test_fps_spread():
    """FPS should spread points — min pairwise distance should be larger than random."""
    from eval.fps import fps_2d
    # Grid of points: FPS should pick well-spread subset
    xs = torch.linspace(0, 1, 20)
    ys = torch.linspace(0, 1, 20)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing='ij'), dim=-1).reshape(-1, 2)  # 400 pts
    result = fps_2d(grid, 20, seed=0)
    dists = torch.cdist(result, result)
    dists.fill_diagonal_(float('inf'))
    min_dist = dists.min().item()
    # On a [0,1]² grid, 20 FPS points should have min spacing > 0.15
    assert min_dist > 0.15, f"FPS min spacing {min_dist} too small"


def test_fps_empty_returns_empty():
    """Empty input returns empty output."""
    from eval.fps import fps_2d
    pts = torch.zeros(0, 2)
    result = fps_2d(pts, 10, seed=0)
    assert result.shape == (0, 2)
