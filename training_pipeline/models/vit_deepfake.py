"""
VeritasAI – Vision Transformer (ViT / Swin) Deepfake Classifier
Uses the `timm` library for state-of-the-art transformer architectures.

Vision Transformers excel at capturing global context (e.g., lighting
inconsistencies across an entire face) that CNNs often miss.

Usage:
    from models.vit_deepfake import build_vit, build_swin
    model = build_vit(pretrained=True, image_size=384)
    model = build_swin(pretrained=True, image_size=384)
"""

import torch
import torch.nn as nn

try:
    import timm

    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


class ViTDeepfakeDetector(nn.Module):
    """Vision Transformer wrapper for binary deepfake classification."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_384",
        pretrained: bool = True,
        image_size: int = 384,
    ):
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError(
                "timm library is required for ViT models. "
                "Install with: pip install timm"
            )

        # Load pretrained ViT backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,   # Remove default classification head
            img_size=image_size,
        )

        # Get the embedding dimension from the backbone
        embed_dim = self.backbone.num_features

        # Custom classifier head for deepfake detection
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(256, 1),
        )

        # Freeze early transformer blocks for efficient fine-tuning
        # Keep the last 4 blocks + classifier trainable
        blocks = list(self.backbone.blocks) if hasattr(self.backbone, "blocks") else []
        if len(blocks) > 4:
            for block in blocks[:-4]:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings without classification (for ensembling)."""
        return self.backbone(x)


class SwinDeepfakeDetector(nn.Module):
    """Swin Transformer wrapper for binary deepfake classification.

    Swin Transformers use shifted windows for better local + global
    feature extraction, making them excellent for detecting subtle
    manipulation artifacts.
    """

    def __init__(
        self,
        model_name: str = "swin_base_patch4_window12_384",
        pretrained: bool = True,
        image_size: int = 384,
    ):
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError(
                "timm library is required for Swin models. "
                "Install with: pip install timm"
            )

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=image_size,
        )

        embed_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings without classification."""
        return self.backbone(x)


def build_vit(
    pretrained: bool = True,
    image_size: int = 384,
    model_name: str = "vit_base_patch16_384",
) -> nn.Module:
    """Create a ViT-based deepfake detector.

    Args:
        pretrained: Use ImageNet-21k pretrained weights.
        image_size: Input resolution (recommended: 384).
        model_name: timm model name (default: vit_base_patch16_384).

    Returns:
        nn.Module with single logit output.
    """
    return ViTDeepfakeDetector(
        model_name=model_name,
        pretrained=pretrained,
        image_size=image_size,
    )


def build_swin(
    pretrained: bool = True,
    image_size: int = 384,
    model_name: str = "swin_base_patch4_window12_384",
) -> nn.Module:
    """Create a Swin Transformer-based deepfake detector.

    Args:
        pretrained: Use ImageNet-21k pretrained weights.
        image_size: Input resolution (recommended: 384).
        model_name: timm model name.

    Returns:
        nn.Module with single logit output.
    """
    return SwinDeepfakeDetector(
        model_name=model_name,
        pretrained=pretrained,
        image_size=image_size,
    )
