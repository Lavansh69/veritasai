"""
VeritasAI – XceptionNet Deepfake Classifier
Depthwise-separable convolution network for deepfake detection.
"""

import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution."""

    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, kernel_size, padding=padding, groups=in_c, bias=False)
        self.pointwise = nn.Conv2d(in_c, out_c, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class XceptionBlock(nn.Module):
    """Residual block with separable convolutions."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.sep1 = SeparableConv2d(in_c, out_c)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.sep2 = SeparableConv2d(out_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, stride=stride, padding=1) if stride > 1 else nn.Identity()
        self.skip = (
            nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm2d(out_c))
            if in_c != out_c or stride > 1
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(self.sep1(x)))
        out = self.bn2(self.sep2(out))
        out = self.pool(out)
        return self.relu(out + residual)


class XceptionNet(nn.Module):
    """Xception-style network for binary deepfake classification."""

    def __init__(self):
        super().__init__()
        # Entry flow
        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Entry blocks
        self.entry_blocks = nn.Sequential(
            XceptionBlock(64, 128, stride=2),
            XceptionBlock(128, 256, stride=2),
            XceptionBlock(256, 728, stride=2),
        )

        # Middle flow (8 repeated blocks)
        self.middle = nn.Sequential(
            *[XceptionBlock(728, 728) for _ in range(8)]
        )

        # Exit flow
        self.exit_block = XceptionBlock(728, 1024, stride=2)
        self.exit_conv = nn.Sequential(
            SeparableConv2d(1024, 1536),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeparableConv2d(1536, 2048),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.entry(x)
        x = self.entry_blocks(x)
        x = self.middle(x)
        x = self.exit_block(x)
        x = self.exit_conv(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_xceptionnet() -> nn.Module:
    """Create an XceptionNet model for deepfake detection."""
    return XceptionNet()
