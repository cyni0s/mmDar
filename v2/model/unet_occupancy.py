"""Symmetric U-Net for polar occupancy prediction.

Takes (B, C_in, 256, 512) feature maps and outputs (B, 1, 256, 512) occupancy.
No asymmetric azimuth upsampling (unlike baseline UNet1 which does 64→512).
Uses same building blocks from train_test_utils/unet_parts.py.
"""
import torch.nn as nn
from train_test_utils.unet_parts import DoubleConv, Down, Up, OutConv


class UNetOcc(nn.Module):
    """Symmetric 4-level U-Net for polar occupancy.

    Architecture:
        Encoder: inc(C→64) → down1(64→128) → down2(128→256) → down3(256→512) → down4(512→512)
        Decoder: up1(1024→256) → up2(512→128) → up3(256→64) → up4(128→64) → outc(64→1) → sigmoid

    Args:
        n_channels: input channels (default 41 for 41-frame stacking)
        n_classes: output channels (default 1 for binary occupancy)
        bilinear: use bilinear upsampling (default True, matches baseline)
    """
    def __init__(self, n_channels=41, n_classes=1, bilinear=True):
        super().__init__()
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return self.sigmoid(x)
