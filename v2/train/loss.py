"""Composite training loss for mmDar v2 point cloud prediction.

Combines four loss terms:
    1. Chamfer distance (PyTorch3D, bidirectional) — primary reconstruction loss
    2. DCD (Density-aware Chamfer Distance) — penalizes non-uniform density
    3. Coverage loss — ensures all GT points have a nearby predicted point
    4. Confidence loss — entropy regularization for per-point confidence

Loss weights:
    total = chamfer + dcd_weight(epoch) * dcd + 0.1 * coverage + 0.01 * confidence

DCD annealing schedule:
    epoch < 5:        weight = 0.0  (Chamfer-only warmup)
    5 <= epoch < 15:  weight = 0.1 * (epoch - 5) / 10.0  (linear ramp)
    epoch >= 15:      weight = 0.1  (full DCD)

References:
    - PyTorch3D chamfer_distance API: https://pytorch3d.readthedocs.io/
    - DCD: "Density-aware Chamfer Distance" (wutong16/Density_aware_Chamfer_Distance)
    - Plan 03-01: composite loss spec
"""

try:
    from pytorch3d.loss import chamfer_distance as _chamfer_distance
    from pytorch3d.ops import knn_points
except ImportError as e:
    raise ImportError(
        "pytorch3d not installed. Run: "
        "pip install git+https://github.com/facebookresearch/pytorch3d.git@stable"
    ) from e

import torch
import torch.nn.functional as F


def chamfer_loss(
    pred_pts: torch.Tensor,
    gt_pts: torch.Tensor,
    weights_x: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bidirectional Chamfer distance loss via PyTorch3D.

    When weights_x is provided, the pred->gt direction applies per-point
    confidence weighting (manually weighted mean). The gt->pred direction
    is always uniformly weighted.

    Args:
        pred_pts:  (B, N, 3) float32 predicted point cloud
        gt_pts:    (B, M, 3) float32 ground-truth point cloud
        weights_x: (B, N) float32 optional per-point confidence weights in [0, 1]
                   applied to the pred->gt direction

    Returns:
        Scalar tensor: weighted_loss_pred_to_gt + loss_gt_to_pred
    """
    if weights_x is None:
        # Standard bidirectional chamfer — pytorch3d handles both directions
        # Returns (loss_xyz, loss_normals); loss_normals is None when no normals
        loss_xyz, _loss_normals = _chamfer_distance(
            pred_pts,
            gt_pts,
            point_reduction="mean",
            batch_reduction="mean",
        )
        return loss_xyz
    else:
        # Weighted pred->gt: use knn to get per-point distances, then weight manually
        knn_xp = knn_points(pred_pts, gt_pts, K=1)
        dist_pred_to_gt = knn_xp.dists.squeeze(-1)  # (B, N) squared distances

        # Weighted mean: sum(w * d) / sum(w) per batch, then mean over batch
        w = weights_x  # (B, N)
        loss_pred_to_gt = (w * dist_pred_to_gt).sum(1) / (w.sum(1) + 1e-8)  # (B,)
        loss_pred_to_gt = loss_pred_to_gt.mean()

        # gt->pred direction (unweighted): compute by swapping x and y
        loss_gt_to_pred_xyz, _normals = _chamfer_distance(
            gt_pts,
            pred_pts,
            single_directional=True,
            point_reduction="mean",
            batch_reduction="mean",
        )
        return loss_pred_to_gt + loss_gt_to_pred_xyz


def dcd_loss(
    pred_pts: torch.Tensor,
    gt_pts: torch.Tensor,
    alpha: float = 1000.0,
) -> torch.Tensor:
    """Density-aware Chamfer Distance (DCD) loss.

    Penalizes over-clustering by weighting each predicted point inversely
    by the number of predictions mapping to the same GT neighbor. This
    encourages uniform coverage instead of cluster-collapse to dense GT regions.

    Inline implementation based on wutong16/Density_aware_Chamfer_Distance.
    Uses pytorch3d.ops.knn_points for efficient batched nearest-neighbor lookup.

    Args:
        pred_pts: (B, N, 3) float32 predicted points
        gt_pts:   (B, M, 3) float32 ground-truth points
        alpha:    Temperature for exponential density weighting (default 1000.0)

    Returns:
        Scalar tensor: mean bidirectional DCD loss over batch
    """
    B, N, _ = pred_pts.shape
    M = gt_pts.shape[1]

    # --- pred -> gt direction ---
    knn_pg = knn_points(pred_pts, gt_pts, K=1)
    dist_pg = knn_pg.dists.squeeze(-1)   # (B, N)  squared distances
    idx_pg = knn_pg.idx.squeeze(-1)      # (B, N)  nearest GT index

    # Soft exponential density weight
    exp_dist_pg = torch.exp(-alpha * dist_pg)  # (B, N)

    # Sum exp weights per GT point
    denom_pg = torch.zeros(B, M, device=pred_pts.device, dtype=pred_pts.dtype)
    denom_pg.scatter_add_(1, idx_pg, exp_dist_pg)

    # Weight for each pred point = exp_dist / sum_of_exp_dists_at_same_GT
    weight_pg = exp_dist_pg / (denom_pg.gather(1, idx_pg) + 1e-8)
    loss_pg = (weight_pg * dist_pg).sum(1) / (weight_pg.sum(1) + 1e-8)  # (B,)

    # --- gt -> pred direction ---
    knn_gp = knn_points(gt_pts, pred_pts, K=1)
    dist_gp = knn_gp.dists.squeeze(-1)   # (B, M)
    idx_gp = knn_gp.idx.squeeze(-1)      # (B, M)

    exp_dist_gp = torch.exp(-alpha * dist_gp)  # (B, M)

    denom_gp = torch.zeros(B, N, device=pred_pts.device, dtype=pred_pts.dtype)
    denom_gp.scatter_add_(1, idx_gp, exp_dist_gp)

    weight_gp = exp_dist_gp / (denom_gp.gather(1, idx_gp) + 1e-8)
    loss_gp = (weight_gp * dist_gp).sum(1) / (weight_gp.sum(1) + 1e-8)  # (B,)

    return (loss_pg.mean() + loss_gp.mean()) / 2.0


def coverage_loss(
    pred_pts: torch.Tensor,
    gt_pts: torch.Tensor,
    threshold: float = 0.25,
) -> torch.Tensor:
    """Coverage hinge loss: penalizes GT points with no nearby prediction.

    For each GT point, finds the distance to the nearest predicted point.
    Applies a squared hinge loss: relu(dist - threshold)^2.

    This loss is zero when all GT points have a predicted point within
    `threshold` meters, and grows quadratically for uncovered GT regions.

    Args:
        pred_pts:  (B, N, 3) float32 predicted points
        gt_pts:    (B, M, 3) float32 ground-truth points
        threshold: Distance threshold in meters (default 0.25)

    Returns:
        Scalar tensor: mean squared hinge loss over all GT points and batch
    """
    # knn: for each GT point, find nearest pred point
    knn_gp = knn_points(gt_pts, pred_pts, K=1)
    dist_gp = knn_gp.dists.squeeze(-1)  # (B, M) squared distances

    # Squared hinge: relu(dist - threshold)^2
    # Note: dist_gp is squared distance, threshold is linear -> compare sqrt
    dist_linear = torch.sqrt(dist_gp + 1e-8)  # (B, M) linear distance
    hinge = F.relu(dist_linear - threshold) ** 2

    return hinge.mean()


def confidence_loss(conf_logits: torch.Tensor) -> torch.Tensor:
    """Entropy regularization pushing per-point confidence toward high confidence.

    Penalizes low-confidence predictions by maximizing log-probability.
    High-confidence predictions (sigmoid -> 1.0) minimize this loss.

    Args:
        conf_logits: (B, N, 1) float32 raw logits (pre-sigmoid)

    Returns:
        Scalar tensor: -mean(log(sigmoid(conf_logits) + eps))
    """
    conf = torch.sigmoid(conf_logits)
    return -torch.mean(torch.log(conf + 1e-8))


def dcd_weight_schedule(epoch: int) -> float:
    """DCD loss weight annealing schedule.

    Warmup phase (epochs 0-4): DCD weight = 0.0 (Chamfer-only learning)
    Ramp phase (epochs 5-14):  DCD weight ramps linearly from 0.0 to 0.1
    Plateau (epoch >= 15):     DCD weight = 0.1 (full DCD contribution)

    Args:
        epoch: Current training epoch (0-indexed)

    Returns:
        float: DCD weight in [0.0, 0.1]
    """
    if epoch < 5:
        return 0.0
    elif epoch < 15:
        return 0.1 * (epoch - 5) / 10.0
    else:
        return 0.1


def composite_loss(
    pred_pts: torch.Tensor,
    gt_pts: torch.Tensor,
    conf_logits: torch.Tensor,
    epoch: int,
    use_dcd: bool = True,
    use_coverage: bool = True,
    use_confidence: bool = True,
    coverage_threshold: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Composite point cloud reconstruction loss.

    Combines Chamfer + DCD + coverage + confidence losses with the DCD
    annealing schedule. Confidence weights are applied to the Chamfer pred->gt
    direction when use_confidence=True.

    Total loss:
        total = chamfer + dcd_weight(epoch) * dcd + 0.1 * coverage + 0.01 * confidence

    Args:
        pred_pts:          (B, N, 3) float32 predicted point cloud
        gt_pts:            (B, M, 3) float32 ground-truth point cloud
        conf_logits:       (B, N, 1) float32 confidence logits (raw, pre-sigmoid)
        epoch:             Current epoch for DCD weight schedule
        use_dcd:           Include DCD loss (default True)
        use_coverage:      Include coverage loss (default True)
        use_confidence:    Include confidence loss + weighted Chamfer (default True)
        coverage_threshold: Coverage hinge threshold in meters (default 0.25)

    Returns:
        Dict with keys: 'total', 'chamfer', 'dcd', 'coverage', 'confidence'
        Each value is a scalar tensor (retains grad_fn for backprop through total).
    """
    # Confidence weights for Chamfer (pred->gt direction)
    weights_x = None
    if use_confidence:
        weights_x = torch.sigmoid(conf_logits).squeeze(-1)  # (B, N)

    # --- Chamfer loss ---
    ch_loss = chamfer_loss(pred_pts, gt_pts, weights_x=weights_x)

    # --- DCD loss ---
    dcd_w = dcd_weight_schedule(epoch)
    if use_dcd and dcd_w > 0.0:
        dc_loss = dcd_loss(pred_pts, gt_pts)
    else:
        dc_loss = torch.tensor(0.0, device=pred_pts.device, dtype=pred_pts.dtype)

    # --- Coverage loss ---
    if use_coverage:
        cov_loss = coverage_loss(pred_pts, gt_pts, threshold=coverage_threshold)
    else:
        cov_loss = torch.tensor(0.0, device=pred_pts.device, dtype=pred_pts.dtype)

    # --- Confidence loss ---
    if use_confidence:
        conf_l = confidence_loss(conf_logits)
    else:
        conf_l = torch.tensor(0.0, device=pred_pts.device, dtype=pred_pts.dtype)

    # --- Total loss ---
    total = ch_loss + dcd_w * dc_loss + 0.1 * cov_loss + 0.01 * conf_l

    return {
        "total": total,
        "chamfer": ch_loss,
        "dcd": dc_loss,
        "coverage": cov_loss,
        "confidence": conf_l,
    }
