"""Direct point cloud evaluation adapter for mmDar v2.

Computes Chamfer distance and modified Hausdorff distance on raw (N, 3) point
clouds WITHOUT going through the polar PNG conversion pipeline. Metrics use
only XY columns (2D projection) to match eval/eval_pointcloud.py's
legacy_cartesian mode exactly.

Metric definitions (matching eval/eval_pointcloud.py and eval/pc_distance.m):
    Chamfer:            0.5 * mean(nn_A->B) + 0.5 * mean(nn_B->A)
    Modified Hausdorff: max(median(nn_A->B), median(nn_B->A))

CRITICAL: XY-only distances (columns [:, :2]) match the legacy_cartesian mode
of the existing eval pipeline. The z coordinate is ignored because the baseline
eval converts polar images to 2D point clouds (no elevation), and our decoder
uses a flat ground prior (z=0 in template). Consistent XY-only distances ensure
fair comparison with the 0.295m Chamfer baseline.

Functions:
    chamfer_distance_np   — (N, 3) x (M, 3) numpy -> float
    mod_hausdorff_np      — (N, 3) x (M, 3) numpy -> float
    evaluate_batch        — (B, N, 3) x (B, M, 3) numpy -> dict
    evaluate_epoch        — model x dataloader -> dict  (requires torch)
"""

import numpy as np
from scipy.spatial.distance import cdist


def chamfer_distance_np(pred: np.ndarray, gt: np.ndarray) -> float:
    """Chamfer distance using only XY columns (2D Euclidean).

    Formula: 0.5 * mean(min_{b in gt} dist(a, b)) + 0.5 * mean(min_{a in pred} dist(b, a))

    Matches eval/eval_pointcloud.py chamfer_distance() with legacy_cartesian mode
    (which operates on 2D points derived from polar images, having no z-column).

    Args:
        pred: (N, 3) float32 or float64 predicted point cloud
        gt:   (M, 3) float32 or float64 ground-truth point cloud

    Returns:
        Chamfer distance in meters (scalar float)
    """
    D = cdist(pred[:, :2], gt[:, :2])  # XY only, (N, M) Euclidean distances
    d_a2b = D.min(axis=1).mean()       # mean nearest-neighbour: pred -> gt
    d_b2a = D.min(axis=0).mean()       # mean nearest-neighbour: gt -> pred
    return float(0.5 * d_a2b + 0.5 * d_b2a)


def mod_hausdorff_np(pred: np.ndarray, gt: np.ndarray) -> float:
    """Modified Hausdorff distance using only XY columns (2D Euclidean).

    Formula: max(median(nn_pred->gt), median(nn_gt->pred))

    Matches eval/eval_pointcloud.py modified_hausdorff() with legacy_cartesian mode.

    Args:
        pred: (N, 3) float32 or float64 predicted point cloud
        gt:   (M, 3) float32 or float64 ground-truth point cloud

    Returns:
        Modified Hausdorff distance in meters (scalar float)
    """
    D = cdist(pred[:, :2], gt[:, :2])   # XY only
    d_pred2gt = np.median(D.min(axis=1))
    d_gt2pred = np.median(D.min(axis=0))
    return float(max(d_pred2gt, d_gt2pred))


def evaluate_batch(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
) -> dict:
    """Evaluate Chamfer and mod-Hausdorff on a batch of point cloud pairs.

    Args:
        pred_pts: (B, N, 3) float32 predicted point clouds
        gt_pts:   (B, M, 3) float32 ground-truth point clouds

    Returns:
        dict with keys:
            'chamfer':      float — mean Chamfer distance over batch
            'mod_hausdorff': float — mean modified Hausdorff over batch
    """
    B = pred_pts.shape[0]
    chamfer_scores = []
    hausdorff_scores = []

    for i in range(B):
        pred_i = pred_pts[i]  # (N, 3)
        gt_i = gt_pts[i]      # (M, 3)
        chamfer_scores.append(chamfer_distance_np(pred_i, gt_i))
        hausdorff_scores.append(mod_hausdorff_np(pred_i, gt_i))

    return {
        "chamfer": float(np.mean(chamfer_scores)),
        "mod_hausdorff": float(np.mean(hausdorff_scores)),
    }


def _chamfer_torch(pred: 'torch.Tensor', gt: 'torch.Tensor') -> float:
    """GPU-accelerated Chamfer distance using only XY columns.

    Uses torch.cdist with chunking for memory efficiency.
    """
    import torch
    pred_xy = pred[:, :2]  # (N, 2)
    gt_xy = gt[:, :2]      # (M, 2)

    # Chunked cdist to avoid OOM on large point clouds
    CHUNK = 2048
    N = pred_xy.shape[0]

    # pred->gt direction
    nn_dists_pg = []
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        d = torch.cdist(pred_xy[start:end], gt_xy)  # (chunk, M)
        nn_dists_pg.append(d.min(dim=1).values)
    d_a2b = torch.cat(nn_dists_pg).mean()

    # gt->pred direction
    M = gt_xy.shape[0]
    nn_dists_gp = []
    for start in range(0, M, CHUNK):
        end = min(start + CHUNK, M)
        d = torch.cdist(gt_xy[start:end], pred_xy)
        nn_dists_gp.append(d.min(dim=1).values)
    d_b2a = torch.cat(nn_dists_gp).mean()

    return float(0.5 * d_a2b + 0.5 * d_b2a)


def _mod_hausdorff_torch(pred: 'torch.Tensor', gt: 'torch.Tensor') -> float:
    """GPU-accelerated modified Hausdorff using only XY columns."""
    import torch
    pred_xy = pred[:, :2]
    gt_xy = gt[:, :2]

    CHUNK = 2048
    N, M = pred_xy.shape[0], gt_xy.shape[0]

    nn_pg = []
    for s in range(0, N, CHUNK):
        e = min(s + CHUNK, N)
        nn_pg.append(torch.cdist(pred_xy[s:e], gt_xy).min(dim=1).values)
    nn_pg = torch.cat(nn_pg)

    nn_gp = []
    for s in range(0, M, CHUNK):
        e = min(s + CHUNK, M)
        nn_gp.append(torch.cdist(gt_xy[s:e], pred_xy).min(dim=1).values)
    nn_gp = torch.cat(nn_gp)

    return float(max(nn_pg.median(), nn_gp.median()))


def evaluate_epoch(model, dataloader, device) -> dict:
    """Run model in eval mode and compute Chamfer + mod-Hausdorff over full epoch.

    Uses GPU-accelerated torch.cdist instead of CPU scipy.cdist for speed.
    Metrics use XY-only 2D distances matching legacy_cartesian baseline.

    Args:
        model:      PyTorch model with forward(radar) -> (pred_pts, conf)
        dataloader: DataLoader yielding (radar, lidar, norm_factor) tuples
        device:     torch.device for model inference

    Returns:
        dict with 'chamfer', 'mod_hausdorff', 'n_samples'
    """
    import torch

    model.eval()
    chamfer_accum = 0.0
    hausdorff_accum = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            radar, lidar, _norm = batch
            radar = radar.to(device)
            lidar = lidar.to(device)

            pred_pts, _conf = model(radar)

            B = pred_pts.shape[0]
            for i in range(B):
                chamfer_accum += _chamfer_torch(pred_pts[i], lidar[i])
                hausdorff_accum += _mod_hausdorff_torch(pred_pts[i], lidar[i])
            n_samples += B

    if n_samples == 0:
        return {"chamfer": float("nan"), "mod_hausdorff": float("nan"), "n_samples": 0}

    return {
        "chamfer": chamfer_accum / n_samples,
        "mod_hausdorff": hausdorff_accum / n_samples,
        "n_samples": n_samples,
    }
