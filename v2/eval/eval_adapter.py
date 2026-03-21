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


def evaluate_epoch(model, dataloader, device) -> dict:
    """Run model in eval mode and compute Chamfer + mod-Hausdorff over full epoch.

    The model is called with radar input and returns (pred_pts, conf).
    Metrics are accumulated over all batches and averaged at the end.

    Args:
        model:      PyTorch model with forward(radar) -> (pred_pts, conf)
                    pred_pts: (B, N, 3) float32 point cloud
        dataloader: DataLoader yielding (radar, lidar, norm_factor) tuples
                    radar: (B, 8, 512) complex64
                    lidar: (B, 8192, 3) float32
        device:     torch.device for model inference

    Returns:
        dict with keys:
            'chamfer':       float — epoch mean Chamfer distance
            'mod_hausdorff': float — epoch mean modified Hausdorff distance
            'n_samples':     int   — total number of samples evaluated
    """
    import torch

    model.eval()
    chamfer_accum = 0.0
    hausdorff_accum = 0.0
    n_batches = 0
    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            radar, lidar, _norm = batch
            radar = radar.to(device)

            pred_pts, _conf = model(radar)

            # Convert to numpy for metric computation
            pred_np = pred_pts.cpu().numpy()   # (B, N, 3)
            gt_np = lidar.cpu().numpy()        # (B, M, 3)

            batch_metrics = evaluate_batch(pred_np, gt_np)
            B = pred_np.shape[0]
            chamfer_accum += batch_metrics["chamfer"] * B
            hausdorff_accum += batch_metrics["mod_hausdorff"] * B
            n_samples += B
            n_batches += 1

    if n_samples == 0:
        return {"chamfer": float("nan"), "mod_hausdorff": float("nan"), "n_samples": 0}

    return {
        "chamfer": chamfer_accum / n_samples,
        "mod_hausdorff": hausdorff_accum / n_samples,
        "n_samples": n_samples,
    }
