"""Loss functions for Gaussian set prediction.

Combines:
1. Hungarian-matched heteroscedastic NLL to GT prototypes
2. Soft coverage loss against full GT cloud
3. Cardinality loss (predicted count ≈ GT count)
4. Repulsion loss (prevent duplicate Gaussians)
5. Sigma prior loss (regularize uncertainty predictions)
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def hungarian_nll_loss(
    mu_xy: torch.Tensor,      # (B, K, 2) predicted centers
    sigma_r: torch.Tensor,    # (B, K) range uncertainty
    sigma_perp: torch.Tensor, # (B, K) perpendicular uncertainty
    mu_r: torch.Tensor,       # (B, K) predicted range (for rotating to local frame)
    mu_phi: torch.Tensor,     # (B, K) predicted azimuth
    existence: torch.Tensor,  # (B, K) existence logits
    gt_xy: torch.Tensor,      # (B, M, 2) GT prototype centers
) -> dict:
    """Hungarian-matched heteroscedastic NLL.

    For each sample, compute cost matrix between K predictions and M GT prototypes,
    solve assignment, then compute NLL only on matched pairs.
    Unmatched predictions contribute to the existence loss.
    """
    B, K, _ = mu_xy.shape
    M = gt_xy.shape[1]
    device = mu_xy.device

    total_nll = torch.tensor(0.0, device=device)
    total_exist = torch.tensor(0.0, device=device)

    for b in range(B):
        # Cost matrix: pairwise L1 distance (for assignment only, not loss)
        with torch.no_grad():
            cost = torch.cdist(mu_xy[b], gt_xy[b], p=1)  # (K, M)
            # Add existence penalty for unconfident predictions
            exist_cost = -torch.sigmoid(existence[b]).unsqueeze(1).expand_as(cost)
            cost = cost + 0.5 * exist_cost
            # Hungarian assignment
            row_idx, col_idx = linear_sum_assignment(cost.cpu().numpy())
            row_idx = torch.tensor(row_idx, device=device)
            col_idx = torch.tensor(col_idx, device=device)

        # Matched pairs: compute heteroscedastic NLL
        matched_pred = mu_xy[b, row_idx]    # (N_match, 2)
        matched_gt = gt_xy[b, col_idx]      # (N_match, 2)
        matched_sr = sigma_r[b, row_idx]    # (N_match,)
        matched_sp = sigma_perp[b, row_idx] # (N_match,)
        matched_r = mu_r[b, row_idx]        # (N_match,)
        matched_phi = mu_phi[b, row_idx]    # (N_match,)

        # Decompose error in local (radial, perpendicular) frame
        delta = matched_gt - matched_pred  # (N_match, 2)
        cos_phi = torch.cos(matched_phi)
        sin_phi = torch.sin(matched_phi)
        dr = delta[:, 0] * cos_phi + delta[:, 1] * sin_phi      # radial error
        dp = -delta[:, 0] * sin_phi + delta[:, 1] * cos_phi     # perp error

        # Heteroscedastic NLL: 0.5*(dr/σ_r)² + 0.5*(dp/σ_p)² + log(σ_r) + log(σ_p)
        nll = 0.5 * (dr / matched_sr) ** 2 + \
              0.5 * (dp / matched_sp) ** 2 + \
              torch.log(matched_sr) + torch.log(matched_sp)
        total_nll = total_nll + nll.mean()

        # Existence loss: matched = 1, unmatched = 0
        exist_target = torch.zeros(K, device=device)
        exist_target[row_idx] = 1.0
        total_exist = total_exist + F.binary_cross_entropy_with_logits(
            existence[b], exist_target
        )

    return {
        'nll': total_nll / B,
        'existence': total_exist / B,
    }


def soft_coverage_loss(
    mu_xy: torch.Tensor,      # (B, K, 2)
    existence: torch.Tensor,  # (B, K) logits
    gt_full: torch.Tensor,    # (B, N, 2) or (B, N, 3) full GT cloud
    tau: float = 0.1,
) -> torch.Tensor:
    """Soft coverage: every GT point should have a nearby confident prediction.

    Uses soft nearest neighbor: d_gp[j] = -τ * logsumexp(-D[:,j]/τ + log_p)
    where log_p = log(sigmoid(existence)) weights by confidence.

    This encourages the model to COVER all GT points, complementing the
    Hungarian NLL which only matches to prototypes.
    """
    B = mu_xy.shape[0]
    gt_xy = gt_full[:, :, :2] if gt_full.shape[-1] > 2 else gt_full

    total = torch.tensor(0.0, device=mu_xy.device)
    for b in range(B):
        D = torch.cdist(mu_xy[b], gt_xy[b])  # (K, N)
        log_p = F.logsigmoid(existence[b])    # (K,)
        # Soft NN from GT to pred: for each GT point, soft-min over predictions
        # weighted by existence probability
        weighted_D = D / tau - log_p.unsqueeze(1)  # (K, N)
        soft_nn_gp = -tau * torch.logsumexp(-weighted_D, dim=0)  # (N,)
        # Mean of soft NN distances (coverage measure)
        total = total + soft_nn_gp.mean()

    return total / B


def cardinality_loss(
    existence: torch.Tensor,  # (B, K) logits
    n_gt: torch.Tensor,       # (B,) number of GT prototypes per sample
) -> torch.Tensor:
    """Penalize mismatch between predicted and GT object count."""
    pred_count = torch.sigmoid(existence).sum(dim=1)  # (B,)
    return F.smooth_l1_loss(pred_count, n_gt.float())


def repulsion_loss(
    mu_xy: torch.Tensor,      # (B, K, 2)
    existence: torch.Tensor,  # (B, K) logits
    rho: float = 0.3,
) -> torch.Tensor:
    """Prevent duplicate Gaussians from collapsing to same location.

    L_rep = mean_{i<k} p_i * p_k * exp(-||μ_i - μ_k||² / (2ρ²))
    """
    B, K, _ = mu_xy.shape
    p = torch.sigmoid(existence)  # (B, K)
    D_sq = torch.cdist(mu_xy, mu_xy) ** 2  # Wait, cdist returns distances, not squared
    D = torch.cdist(mu_xy, mu_xy)  # (B, K, K)
    D_sq = D ** 2

    # Weight by existence probability of both Gaussians
    p_outer = p.unsqueeze(2) * p.unsqueeze(1)  # (B, K, K)
    repel = p_outer * torch.exp(-D_sq / (2 * rho ** 2))

    # Only upper triangle (i < k), excluding diagonal
    mask = torch.triu(torch.ones(K, K, device=mu_xy.device), diagonal=1).bool()
    return repel[:, mask].mean()


def sigma_prior_loss(
    sigma_r: torch.Tensor,      # (B, K)
    sigma_perp: torch.Tensor,   # (B, K)
    sigma_r_prior: float = 0.1,
    sigma_perp_prior: float = 0.3,
) -> torch.Tensor:
    """Regularize uncertainties toward physics-informed priors.

    Penalizes deviation from expected uncertainty scale.
    """
    loss_r = (torch.log(sigma_r) - math.log(sigma_r_prior)) ** 2
    loss_p = (torch.log(sigma_perp) - math.log(sigma_perp_prior)) ** 2
    return (loss_r.mean() + loss_p.mean()) * 0.5


import math


def gaussian_composite_loss(
    model_out: dict,
    gt_prototypes: torch.Tensor,  # (B, M, 2) GT prototype centers
    gt_full: torch.Tensor,        # (B, N, 2) or (B, N, 3) full GT cloud
    n_gt: torch.Tensor,           # (B,) number of GT prototypes
    epoch: int = 0,
    # Loss weights
    w_nll: float = 1.0,
    w_exist: float = 1.0,
    w_coverage: float = 0.5,
    w_cardinality: float = 0.5,
    w_repulsion: float = 0.1,
    w_sigma_prior: float = 0.1,
) -> dict:
    """Composite loss combining all Gaussian set prediction losses.

    Args:
        model_out: dict from GaussianSetDecoder.forward()
        gt_prototypes: (B, M, 2) K-Means centers of lidar GT
        gt_full: (B, N, 2+) full lidar GT point cloud
        n_gt: (B,) number of valid GT prototypes per sample
        epoch: current epoch (for scheduling)

    Returns:
        dict with 'total' and individual loss components
    """
    mu_xy = model_out['mu_xy']
    mu_r = model_out['mu_r']
    mu_phi = model_out['mu_phi']
    sigma_r = model_out['sigma_r']
    sigma_perp = model_out['sigma_perp']
    existence = model_out['existence']

    # 1. Hungarian NLL + existence
    hun = hungarian_nll_loss(mu_xy, sigma_r, sigma_perp, mu_r, mu_phi,
                             existence, gt_prototypes)

    # 2. Soft coverage against full GT
    cov = soft_coverage_loss(mu_xy, existence, gt_full)

    # 3. Cardinality
    card = cardinality_loss(existence, n_gt)

    # 4. Repulsion
    rep = repulsion_loss(mu_xy, existence)

    # 5. Sigma prior
    sig = sigma_prior_loss(sigma_r, sigma_perp)

    # Warm-up: first few epochs focus on position, then add uncertainty
    if epoch < 3:
        w_sigma_prior = 0.0
        # Use larger sigma initially (frozen)

    total = (w_nll * hun['nll'] +
             w_exist * hun['existence'] +
             w_coverage * cov +
             w_cardinality * card +
             w_repulsion * rep +
             w_sigma_prior * sig)

    return {
        'total': total,
        'nll': hun['nll'].detach(),
        'existence': hun['existence'].detach(),
        'coverage': cov.detach(),
        'cardinality': card.detach(),
        'repulsion': rep.detach(),
        'sigma_prior': sig.detach(),
    }
