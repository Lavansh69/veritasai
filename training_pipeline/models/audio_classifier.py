"""
VeritasAI – Audio Deepfake Classifier Model
Mel-spectrogram CNN architecture for detecting AI-generated / cloned audio.
Completely separate from all image/video model files.
"""

import torch
import torch.nn as nn


class AudioClassifier(nn.Module):
    """CNN classifier operating on Mel-spectrograms.

    Architecture:
        Input → 4 Conv blocks → AdaptiveAvgPool → FC → sigmoid

    Input shape:  (batch, 1, n_mels, time_frames)
    Output shape: (batch, 1)  — raw logits, apply sigmoid for probability
    """

    def __init__(self, n_mels: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1 → 32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    # Quick shape sanity check
    model = AudioClassifier()
    dummy = torch.randn(4, 1, 128, 157)  # batch=4, 1ch, 128 mels, ~5s@16kHz
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
