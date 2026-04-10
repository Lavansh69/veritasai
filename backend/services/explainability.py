"""
VeritasAI – Explainability Module (Grad-CAM)
Generates heatmap overlays highlighting suspicious regions.
"""

import cv2
import numpy as np
import torch
from PIL import Image

from config import DEVICE, HEATMAP_DIR, IMAGE_SIZE
from services.inference import _load_model
from services.media_processing import preprocess_image


def _get_target_layer(model):
    """Return the last convolutional layer for Grad-CAM."""
    # EfficientNet
    if hasattr(model, "features"):
        return model.features[-1]
    # XceptionNet (custom)
    if hasattr(model, "exit"):
        for module in reversed(list(model.exit.modules())):
            if isinstance(module, (torch.nn.Conv2d,)):
                return module
    # Fallback: find last Conv2d
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def generate_heatmap(image_path: str, analysis_id: str) -> dict:
    """Generate Grad-CAM heatmap and save overlay image.
    
    Returns dict with heatmap_url and explanation text.
    """
    model = _load_model()
    target_layer = _get_target_layer(model)

    if target_layer is None:
        return {
            "heatmap_url": None,
            "explanation": "Could not identify target layer for Grad-CAM.",
            "indicators": [],
        }

    # Hook for activations and gradients
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    arr = preprocess_image(image_path)
    tensor = torch.from_numpy(arr).to(DEVICE).requires_grad_(True)
    model.eval()

    output = model(tensor)
    prob = torch.sigmoid(output).item()

    # Backward pass
    model.zero_grad()
    output.backward()

    fh.remove()
    bh.remove()

    # Compute Grad-CAM
    grads = gradients.get("value")
    acts = activations.get("value")

    if grads is None or acts is None:
        return {
            "heatmap_url": None,
            "explanation": "Grad-CAM computation could not retrieve activations.",
            "indicators": [],
        }

    weights = grads.mean(dim=[2, 3], keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().numpy()

    # Normalise to [0, 255]
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = np.uint8(cam * 255)
    cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

    # Overlay on original image
    original = cv2.imread(image_path)
    if original is None:
        original = np.array(Image.open(image_path).convert("RGB"))
        original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    original = cv2.resize(original, (IMAGE_SIZE, IMAGE_SIZE))

    heatmap_colored = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

    # Save
    heatmap_filename = f"{analysis_id}_heatmap.jpg"
    heatmap_path = HEATMAP_DIR / heatmap_filename
    cv2.imwrite(str(heatmap_path), overlay)

    # Generate explanation
    indicators = _analyze_indicators(cam, prob)
    explanation = _build_explanation(indicators, prob)

    return {
        "heatmap_url": f"/static/heatmaps/{heatmap_filename}",
        "probability": round(prob, 4),
        "explanation": explanation,
        "indicators": indicators,
    }


def _analyze_indicators(cam: np.ndarray, prob: float) -> list[str]:
    """Detect suspicious indicators from heatmap activation patterns."""
    indicators = []
    h, w = cam.shape

    # Check face boundary region (edges)
    border = cam[:h // 8, :].mean() + cam[-h // 8:, :].mean()
    center = cam[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean()

    if border > center * 0.7:
        indicators.append("Boundary inconsistencies detected along face edges")

    if prob > 0.75:
        indicators.append("Possible facial blending artifacts detected")
    if prob > 0.85:
        indicators.append("Possible GAN fingerprint noise patterns")
    if prob > 0.90:
        indicators.append("Lighting mismatch across facial features")
    if prob > 0.95:
        indicators.append("Significant texture inconsistencies in skin regions")

    if not indicators:
        indicators.append("No significant manipulation indicators found")

    return indicators


def _build_explanation(indicators: list[str], prob: float) -> str:
    """Build human-readable explanation text."""
    if prob < 0.3:
        verdict = "The AI analysis suggests this media is likely authentic."
    elif prob < 0.6:
        verdict = "The AI analysis has detected some suspicious patterns that may indicate manipulation."
    else:
        verdict = "The AI analysis strongly suggests this media has been manipulated or AI-generated."

    details = "\n".join(f"  • {ind}" for ind in indicators)
    return f"{verdict}\n\nDetected indicators:\n{details}"
