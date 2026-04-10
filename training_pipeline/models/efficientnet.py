"""
VeritasAI – EfficientNet-B4 Deepfake Classifier
"""

import torch.nn as nn
from torchvision import models


def build_efficientnet(pretrained: bool = True) -> nn.Module:
    """EfficientNet-B4 adapted for binary deepfake classification.
    
    Args:
        pretrained: Use ImageNet pretrained weights for transfer learning.
    
    Returns:
        nn.Module with single sigmoid output.
    """
    weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b4(weights=weights)

    # Freeze early layers for fine-tuning
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 1),
    )
    return model
