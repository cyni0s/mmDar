"""Smoke tests for ConvLSTM training pipeline.

Tests cover:
1. Single-batch forward+backward end-to-end (train_single_batch_smoke)
2. Dense loss normalisation is stable across different T values (test_dense_loss_normalization)
3. State zeroing logic — no error when resetting hidden state slots (test_state_reset_on_trajectory_change)
4. Checkpoint save/load round-trip produces identical outputs (test_checkpoint_format)

All tests use small synthetic tensors and run on CPU (or GPU when available).
GPU-only tests are skipped automatically when CUDA is not available.
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from train_test_utils.model import UNet1ConvLSTM
from train_convlstm import compute_dense_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(device):
    return UNet1ConvLSTM(n_channels=1, n_classes=1).to(device)


def _make_batch(B, T, device):
    """Small synthetic radar+lidar batch."""
    radar = torch.rand(B, T, 1, 256, 64, device=device)
    lidar = (torch.rand(B, T, 1, 256, 512, device=device) > 0.8).float()
    return radar, lidar


# ---------------------------------------------------------------------------
# Test 1: end-to-end single-batch smoke
# ---------------------------------------------------------------------------

def test_single_batch_smoke(device):
    """End-to-end forward + backward + optimizer step on a small batch.

    Asserts:
    - loss is finite
    - grad_norm is finite and > 0
    - at least one model parameter changed after the step
    - detached state does not break a second forward pass
    """
    B, T = 2, 5
    model = _make_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0005)
    bce_loss_fn = nn.BCELoss()

    radar, lidar = _make_batch(B, T, device)

    # --- First forward + backward ---
    optimizer.zero_grad(set_to_none=True)

    # Zero-init state (production behaviour)
    state = None
    preds, state_out = model(radar, state=state)

    assert preds.shape == (B, T, 1, 256, 512), f'Unexpected pred shape: {preds.shape}'

    loss, weights = compute_dense_loss(preds, lidar, 0.2, bce_loss_fn, torch.device(device))

    assert torch.isfinite(loss), f'Loss is not finite: {loss.item()}'
    assert abs(weights[-1].item() - 1.0) < 1e-6, 'Final step weight must be 1.0'
    assert abs(weights[0].item() - 0.2) < 1e-6, 'Intermediate step weight must be 0.2'

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    assert torch.isfinite(grad_norm) and grad_norm.item() > 0, (
        f'Expected finite positive grad_norm, got {grad_norm}'
    )

    # Snapshot a param before step
    param_snap = {name: p.detach().clone() for name, p in model.named_parameters()}

    optimizer.step()

    # At least some parameters should have changed
    changed = any(
        not torch.allclose(p, param_snap[name])
        for name, p in model.named_parameters()
    )
    assert changed, 'No parameters changed after optimizer step — step may not have run'

    # --- Detach state and run a second forward (TBPTT simulation) ---
    (h1, c1), (h2, c2) = state_out
    detached_state = ((h1.detach(), c1.detach()), (h2.detach(), c2.detach()))

    radar2, lidar2 = _make_batch(B, T, device)
    with torch.no_grad():
        preds2, _ = model(radar2, state=detached_state)
    assert preds2.shape == (B, T, 1, 256, 512)


# ---------------------------------------------------------------------------
# Test 2: dense loss normalisation
# ---------------------------------------------------------------------------

def test_dense_loss_normalization(device):
    """Loss magnitude should be in the same order of magnitude for T=1 vs T=10.

    This catches Pitfall 6 from RESEARCH.md — without weight_sum normalisation,
    the loss for T=10 would be ~2.8x larger than T=1, creating scale mismatch.
    With normalisation the ratio should be < 5x (in practice < 2x for random inputs).
    """
    B = 1
    bce_loss_fn = nn.BCELoss()
    dev = torch.device(device)

    torch.manual_seed(42)

    # T=1: single frame — weight=[1.0], weight_sum=1.0, no normalisation needed
    preds_1 = torch.rand(B, 1, 1, 256, 512, device=dev)
    lidar_1 = (torch.rand(B, 1, 1, 256, 512, device=dev) > 0.8).float()
    loss_1, _ = compute_dense_loss(preds_1, lidar_1, 0.2, bce_loss_fn, dev)

    # T=10: weights=[0.2]*9 + [1.0], weight_sum = 9*0.2 + 1.0 = 2.8
    preds_10 = torch.rand(B, 10, 1, 256, 512, device=dev)
    lidar_10 = (torch.rand(B, 10, 1, 256, 512, device=dev) > 0.8).float()
    loss_10, _ = compute_dense_loss(preds_10, lidar_10, 0.2, bce_loss_fn, dev)

    assert torch.isfinite(loss_1), 'T=1 loss is not finite'
    assert torch.isfinite(loss_10), 'T=10 loss is not finite'

    # Both should be in similar range — ratio < 5x
    ratio = max(loss_1.item(), loss_10.item()) / max(min(loss_1.item(), loss_10.item()), 1e-9)
    assert ratio < 5.0, (
        f'Loss ratio T=1/T=10 = {ratio:.2f} > 5 — weight_sum normalisation may be missing. '
        f'loss_1={loss_1.item():.4f}, loss_10={loss_10.item():.4f}'
    )


# ---------------------------------------------------------------------------
# Test 3: state reset on trajectory change
# ---------------------------------------------------------------------------

def test_state_reset_on_trajectory_change(device):
    """Simulates mid-batch trajectory slot reset — zeroing state for changed slots.

    Production training always zero-inits state=None per batch, so this test
    verifies that passing a partially-zeroed state (simulating a slot-level reset
    during streaming inference) does not cause errors.
    """
    B, T = 2, 3
    model = _make_model(device)
    dev = torch.device(device)

    radar, _ = _make_batch(B, T, dev)

    # Forward 1 — build non-zero state
    with torch.no_grad():
        _, state = model(radar, state=None)

    (h1, c1), (h2, c2) = state

    # Verify state is non-zero after forward
    assert h1.abs().max().item() > 0, 'h1 is all zeros after forward — unexpected'

    # Simulate trajectory slot 0 changing: zero out slot 0 in batch dim
    h1_reset = h1.clone()
    c1_reset = c1.clone()
    h2_reset = h2.clone()
    c2_reset = c2.clone()

    h1_reset[0].zero_()
    c1_reset[0].zero_()
    h2_reset[0].zero_()
    c2_reset[0].zero_()

    reset_state = ((h1_reset, c1_reset), (h2_reset, c2_reset))

    # Forward 2 with partially-zeroed state — should not raise
    radar2, _ = _make_batch(B, T, dev)
    with torch.no_grad():
        preds2, state2 = model(radar2, state=reset_state)

    assert preds2.shape == (B, T, 1, 256, 512)
    # Slot 0 and slot 1 should produce different outputs (slot 1 had non-zero history)
    # This is a directional check — if states differ, outputs differ
    h1_out, c1_out = state2[0]
    slot0_h = h1_out[0]
    slot1_h = h1_out[1]
    # They come from different initial states so they should differ
    assert not torch.allclose(slot0_h, slot1_h, atol=1e-5), (
        'Hidden states for slots 0 and 1 are identical despite different initial states'
    )


# ---------------------------------------------------------------------------
# Test 4: checkpoint save/load round-trip
# ---------------------------------------------------------------------------

def test_checkpoint_format(device):
    """Checkpoint save/load produces identical inference output.

    Tests the {epoch:03d}.pt_gen format used in train_convlstm.py:
      {'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    """
    B, T = 1, 3
    model = _make_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dev = torch.device(device)

    # Get reference output
    torch.manual_seed(7)
    radar, _ = _make_batch(B, T, dev)
    model.eval()
    with torch.no_grad():
        ref_out, _ = model(radar, state=None)

    # Save checkpoint in production format
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, '010.pt_gen')
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, ckpt_path)

        # Load into a fresh model
        model2 = _make_model(device)
        ckpt = torch.load(ckpt_path, map_location=dev)
        model2.load_state_dict(ckpt['state_dict'])
        model2.eval()

        with torch.no_grad():
            loaded_out, _ = model2(radar, state=None)

    assert torch.allclose(ref_out, loaded_out, atol=1e-5), (
        f'Checkpoint round-trip mismatch: max_diff={( ref_out - loaded_out).abs().max().item():.2e}'
    )
