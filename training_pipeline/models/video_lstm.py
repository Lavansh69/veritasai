"""
VeritasAI – Temporal Video Analysis (LSTM / Transformer)
Detects temporal inconsistencies across video frames that frame-by-frame
averaging completely misses: flickering, inconsistent micro-expressions,
unnatural blinking, and temporal blending artifacts.

Architecture:
    1. EfficientNet-B4 backbone extracts per-frame embeddings
    2. Bidirectional LSTM (or TransformerEncoder) processes the sequence
    3. Final MLP outputs a single deepfake probability for the video

Usage:
    from models.video_lstm import VideoLSTM, VideoTransformer
    model = VideoLSTM(backbone_features=1792, hidden_size=512)
"""

import torch
import torch.nn as nn
from torchvision import models


class FrameEncoder(nn.Module):
    """Extracts frame-level embeddings using EfficientNet-B4 backbone.

    This strips the classifier head and returns the penultimate feature
    vector for each frame.
    """

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b4(weights=weights)
        self.num_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) single frame tensor
        Returns:
            (B, num_features) embedding vector
        """
        return self.backbone(x)


class VideoLSTM(nn.Module):
    """Bidirectional LSTM for temporal deepfake detection.

    Processes a sequence of frame embeddings and outputs a single
    probability for the entire video.
    """

    def __init__(
        self,
        backbone_features: int = 1792,  # EfficientNet-B4 feature dim
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # Frame-level feature extractor
        self.frame_encoder = FrameEncoder(
            pretrained=pretrained_backbone, freeze=freeze_backbone
        )

        # Temporal sequence model
        self.lstm = nn.LSTM(
            input_size=backbone_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention mechanism to weight important frames
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W) batch of video frame sequences

        Returns:
            (B, 1) deepfake logit for each video
        """
        B, T, C, H, W = frames.shape

        # Extract per-frame embeddings
        frames_flat = frames.view(B * T, C, H, W)
        embeddings = self.frame_encoder(frames_flat)  # (B*T, F)
        embeddings = embeddings.view(B, T, -1)        # (B, T, F)

        # LSTM temporal processing
        lstm_out, _ = self.lstm(embeddings)  # (B, T, hidden*2)

        # Attention-weighted pooling
        attn_weights = self.attention(lstm_out)        # (B, T, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (B, hidden*2)

        return self.classifier(context)

    def forward_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Run temporal analysis on pre-computed frame embeddings.

        Useful when embeddings are pre-cached to avoid redundant backbone
        computation.

        Args:
            embeddings: (B, T, F) pre-computed frame embeddings

        Returns:
            (B, 1) deepfake logit for each video
        """
        lstm_out, _ = self.lstm(embeddings)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        return self.classifier(context)


class VideoTransformer(nn.Module):
    """Transformer-based temporal analysis for deepfake detection.

    Alternative to LSTM that uses self-attention across frames,
    which may capture longer-range temporal dependencies better.
    """

    def __init__(
        self,
        backbone_features: int = 1792,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.3,
        max_frames: int = 64,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.frame_encoder = FrameEncoder(
            pretrained=pretrained_backbone, freeze=freeze_backbone
        )

        # Project backbone features to transformer dimension
        self.input_projection = nn.Linear(backbone_features, d_model)

        # Learnable positional encoding for frame positions
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_frames, d_model) * 0.02
        )

        # CLS token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classifier on CLS token
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(p=0.3),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(256, 1),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W) batch of video frame sequences

        Returns:
            (B, 1) deepfake logit for each video
        """
        B, T, C, H, W = frames.shape

        # Extract per-frame embeddings
        frames_flat = frames.view(B * T, C, H, W)
        embeddings = self.frame_encoder(frames_flat)
        embeddings = embeddings.view(B, T, -1)

        return self.forward_from_embeddings(embeddings)

    def forward_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Run temporal analysis on pre-computed frame embeddings.

        Args:
            embeddings: (B, T, F) pre-computed frame embeddings

        Returns:
            (B, 1) deepfake logit for each video
        """
        B, T, _ = embeddings.shape

        # Project to transformer dimension
        x = self.input_projection(embeddings)  # (B, T, d_model)

        # Add positional encoding
        x = x + self.pos_encoding[:, :T, :]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)

        # Transformer
        x = self.transformer(x)

        # Use CLS token output for classification
        cls_output = x[:, 0, :]  # (B, d_model)
        return self.classifier(cls_output)


def build_video_lstm(
    pretrained: bool = True, freeze_backbone: bool = True
) -> nn.Module:
    """Create a VideoLSTM model for temporal deepfake detection."""
    return VideoLSTM(
        pretrained_backbone=pretrained,
        freeze_backbone=freeze_backbone,
    )


def build_video_transformer(
    pretrained: bool = True, freeze_backbone: bool = True
) -> nn.Module:
    """Create a VideoTransformer model for temporal deepfake detection."""
    return VideoTransformer(
        pretrained_backbone=pretrained,
        freeze_backbone=freeze_backbone,
    )
