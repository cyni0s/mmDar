"""Farthest Point Sampling on 2D point clouds (pure PyTorch, no Open3D)."""

import torch


def fps_2d(points: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    """Greedy farthest-point sampling on 2D points.

    Args:
        points: (K, 2) tensor of 2D points.
        n: target number of points.
        seed: deterministic start index = seed % K.

    Returns:
        (min(K, n), 2) tensor of selected points.
        Returns (0, 2) if input is empty.
    """
    K = points.shape[0]
    if K == 0:
        return points[:0]  # preserve (0, 2) shape
    if K <= n:
        return points

    selected = [seed % K]
    dists = torch.full((K,), float('inf'), device=points.device)

    for _ in range(n - 1):
        new_dists = torch.cdist(
            points, points[selected[-1]].unsqueeze(0)
        ).squeeze(1)
        dists = torch.minimum(dists, new_dists)
        selected.append(int(dists.argmax()))

    return points[selected]
