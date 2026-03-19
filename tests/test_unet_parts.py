"""Unit tests for norm_type parameter in unet_parts.py."""
import pytest
import torch
import torch.nn as nn

from train_test_utils.unet_parts import DoubleConv, Down, Up, Up_nocat, Up_nocat_sym
from train_test_utils.model import UNet1


def _has_batchnorm(module):
    return any(isinstance(m, nn.BatchNorm2d) for m in module.modules())


def _has_groupnorm(module):
    return any(isinstance(m, nn.GroupNorm) for m in module.modules())


# ---------- DoubleConv tests ----------

def test_doubleconv_batch_contains_batchnorm():
    """DoubleConv with norm_type='batch' must contain BatchNorm2d."""
    m = DoubleConv(64, 128, norm_type='batch')
    assert _has_batchnorm(m), "Expected BatchNorm2d for norm_type='batch'"


def test_doubleconv_group_contains_groupnorm():
    """DoubleConv with norm_type='group' must contain GroupNorm."""
    m = DoubleConv(64, 128, norm_type='group')
    assert _has_groupnorm(m), "Expected GroupNorm for norm_type='group'"
    assert not _has_batchnorm(m), "Should NOT contain BatchNorm2d when norm_type='group'"


def test_doubleconv_group_uses_32_groups():
    """GroupNorm must use num_groups=32."""
    m = DoubleConv(64, 128, norm_type='group')
    gn_layers = [mod for mod in m.modules() if isinstance(mod, nn.GroupNorm)]
    assert len(gn_layers) == 2, f"Expected 2 GroupNorm layers, got {len(gn_layers)}"
    for gn in gn_layers:
        assert gn.num_groups == 32, f"Expected num_groups=32, got {gn.num_groups}"


def test_doubleconv_default_is_batch():
    """DoubleConv() without norm_type must default to batch (backward compat)."""
    m = DoubleConv(64, 128)
    assert _has_batchnorm(m), "Default norm_type must be 'batch' for backward compat"


def test_doubleconv_invalid_norm_raises():
    """Unknown norm_type must raise ValueError."""
    with pytest.raises(ValueError):
        DoubleConv(64, 128, norm_type='invalid_norm')


# ---------- Down tests ----------

def test_down_group_propagates_norm():
    """Down(64, 128, norm_type='group') must propagate norm_type to inner DoubleConv."""
    m = Down(64, 128, norm_type='group')
    assert _has_groupnorm(m), "Expected GroupNorm inside Down with norm_type='group'"
    assert not _has_batchnorm(m), "Should NOT contain BatchNorm2d when norm_type='group'"


def test_down_default_is_batch():
    """Down() default must use BatchNorm2d."""
    m = Down(64, 128)
    assert _has_batchnorm(m), "Default Down must use BatchNorm2d"


# ---------- Up tests ----------

def test_up_group_propagates_norm():
    """Up(1024, 256, bilinear=True, norm_type='group') must propagate norm_type."""
    m = Up(1024, 256, bilinear=True, norm_type='group')
    assert _has_groupnorm(m), "Expected GroupNorm inside Up with norm_type='group'"
    assert not _has_batchnorm(m), "Should NOT contain BatchNorm2d when norm_type='group'"


def test_up_default_is_batch():
    """Up() default must use BatchNorm2d."""
    m = Up(1024, 256, bilinear=True)
    assert _has_batchnorm(m), "Default Up must use BatchNorm2d"


# ---------- Up_nocat tests ----------

def test_up_nocat_group_propagates_norm():
    """Up_nocat with norm_type='group' must propagate norm_type."""
    m = Up_nocat(64, 64, bilinear=True, norm_type='group')
    assert _has_groupnorm(m), "Expected GroupNorm inside Up_nocat with norm_type='group'"
    assert not _has_batchnorm(m), "Should NOT contain BatchNorm2d when norm_type='group'"


def test_up_nocat_default_is_batch():
    """Up_nocat() default must use BatchNorm2d."""
    m = Up_nocat(64, 64, bilinear=True)
    assert _has_batchnorm(m), "Default Up_nocat must use BatchNorm2d"


# ---------- UNet1 regression test ----------

def test_unet1_regression_shape():
    """UNet1(41, 1) forward with (1, 41, 256, 64) must still produce (1, 1, 256, 512)."""
    model = UNet1(41, 1)
    model.eval()
    with torch.no_grad():
        inp = torch.randn(1, 41, 256, 64)
        out = model(inp)
    assert out.shape == (1, 1, 256, 512), f"Expected (1,1,256,512), got {out.shape}"


def test_unet1_regression_no_groupnorm():
    """UNet1(41, 1) must NOT contain GroupNorm (purely BatchNorm baseline)."""
    model = UNet1(41, 1)
    assert not _has_groupnorm(model), "UNet1 baseline should use BatchNorm2d, not GroupNorm"
