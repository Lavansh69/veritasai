"""
VeritasAI – Frequency Domain Fusion Network
Dual-branch model that fuses spatial (RGB) and frequency (DCT/FFT) features
for improved deepfake detection.

AI generators (GANs, Diffusion models) leave distinct high-frequency
artifacts that are invisible in the spatial domain but clearly visible
in the frequency domain. This model captures those patterns.

Usage:
    from models.frequency_fusion_net import build_frequency_fusion_net
    model = build_frequency_fusion_net(pretrained=True, image_size=384)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models


class DCTTransform(nn.Module):
    """Compute 2D Discrete Cosine Transform of an image.

    Converts RGB input into frequency-domain representation that
    highlights GAN fingerprint patterns.
    """

    def __init__(self, image_size: int = 384):
        super().__init__()
        self.image_size = image_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalised RGB tensor
        Returns:
            (B, 3, H, W) frequency spectrum tensor (log-magnitude)
        """
        # Use FFT2 as a fast proxy for DCT (captures same frequency info)
        freq = torch.fft.fft2(x, dim=(-2, -1))
        freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))

        # Log-magnitude spectrum (add small epsilon to avoid log(0))
        magnitude = torch.abs(freq_shifted)
        log_magnitude = torch.log1p(magnitude)

        # Normalise per-sample to [0, 1] range
        B = log_magnitude.shape[0]
        flat = log_magnitude.view(B, -1)
        mins = flat.min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        maxs = flat.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        log_magnitude = (log_magnitude - mins) / (maxs - mins + 1e-8)

        return log_magnitude


class FrequencyBranch(nn.Module):
    """Lightweight CNN to process frequency-domain features."""

    def __init__(self, out_features: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 4: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 5: 256 → 512
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(512, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)


class FrequencyFusionNet(nn.Module):
    """Dual-branch network fusing spatial + frequency features.

    Branch A: EfficientNet-B4 on RGB image → spatial embedding
    Branch B: Lightweight CNN on DCT/FFT spectrum → frequency embedding
    Fusion: Concatenate → MLP → single logit output
    """

    def __init__(self, pretrained: bool = True, image_size: int = 384):
        super().__init__()

        # ── Branch A: Spatial (EfficientNet-B4) ─────────────────────
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        self.spatial_backbone = models.efficientnet_b4(weights=weights)

        # Remove the classifier to get the feature extractor
        spatial_in_features = self.spatial_backbone.classifier[1].in_features
        self.spatial_backbone.classifier = nn.Identity()

        # Freeze early spatial layers for transfer learning
        for param in list(self.spatial_backbone.parameters())[:-30]:
            param.requires_grad = False

        # ── Branch B: Frequency ─────────────────────────────────────
        self.dct_transform = DCTTransform(image_size)
        self.freq_branch = FrequencyBranch(out_features=512)

        # ── Fusion Head ─────────────────────────────────────────────
        fused_dim = spatial_in_features + 512
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch A: spatial features from RGB
        spatial_features = self.spatial_backbone(x)

        # Branch B: frequency features from DCT/FFT
        freq_input = self.dct_transform(x)
        freq_features = self.freq_branch(freq_input)

        # Fuse and classify
        fused = torch.cat([spatial_features, freq_features], dim=1)
        return self.classifier(fused)


def build_frequency_fusion_net(
    pretrained: bool = True, image_size: int = 384
) -> nn.Module:
    """Create a FrequencyFusionNet model for deepfake detection.

    Args:
        pretrained: Use ImageNet pretrained weights for the spatial branch.
        image_size: Input image resolution.

    Returns:
        nn.Module with single logit output.
    """
    return FrequencyFusionNet(pretrained=pretrained, image_size=image_size)
