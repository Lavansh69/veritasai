"""
VeritasAI - Forensic Image Analysis
ELA (Error Level Analysis) + Frequency Analysis for AI-generated image detection.
These methods work on ALL types of AI-generated content without training.

ELA: Re-saves at a known JPEG quality and compares pixel differences.
     AI images show uniform error levels; real photos have varied levels.

Frequency: Analyzes DCT/FFT patterns. AI images lack high-frequency noise 
     that real camera sensors produce.
"""

import logging
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


def analyze_ela(image_path: str, quality: int = 90) -> dict:
    """Error Level Analysis - detects manipulation and AI generation.
    
    AI-generated images produce unusually uniform ELA patterns because
    they don't go through real camera JPEG compression cycles.
    
    Returns dict with ela_score (0-100, higher = more suspicious)
    """
    try:
        original = Image.open(image_path).convert("RGB")
        
        # Re-save at MULTIPLE quality levels and compare
        # This makes ELA more sensitive for WhatsApp images
        scores_per_quality = []
        for q in [75, 85, 95]:
            buffer = BytesIO()
            original.save(buffer, format="JPEG", quality=q)
            buffer.seek(0)
            resaved = Image.open(buffer).convert("RGB")
            
            orig_arr = np.array(original, dtype=np.float32)
            resv_arr = np.array(resaved, dtype=np.float32)
            diff = np.abs(orig_arr - resv_arr)
            scores_per_quality.append(diff)
        
        # Average across quality levels
        avg_diff = np.mean(scores_per_quality, axis=0)
        
        scale_factor = 20.0
        ela = np.clip(avg_diff * scale_factor, 0, 255).astype(np.uint8)
        ela_gray = cv2.cvtColor(ela, cv2.COLOR_RGB2GRAY)
        
        mean_ela = float(np.mean(ela_gray))
        std_ela = float(np.std(ela_gray))
        
        if mean_ela > 0:
            uniformity = std_ela / mean_ela
        else:
            uniformity = 1.0
        
        low_ela_ratio = float(np.mean(ela_gray < 5))
        mid_ela_ratio = float(np.mean((ela_gray >= 10) & (ela_gray <= 80)))
        
        # Analyze block-level variance (8x8 JPEG blocks)
        h, w = ela_gray.shape
        block_vars = []
        for by in range(0, h - 7, 8):
            for bx in range(0, w - 7, 8):
                block = ela_gray[by:by+8, bx:bx+8].astype(float)
                block_vars.append(np.var(block))
        block_var_std = float(np.std(block_vars)) if block_vars else 0
        mean_block_var = float(np.mean(block_vars)) if block_vars else 0
        
        score = 0.0
        
        # Block variance analysis: AI images have more uniform blocks
        if mean_block_var > 0 and block_var_std / mean_block_var < 0.8:
            score += 20
        
        if uniformity < 0.4:
            score += 35
        elif uniformity < 0.6:
            score += 25
        elif uniformity < 0.8:
            score += 15
        
        # High percentage of mid-range ELA = suspicious
        if mid_ela_ratio > 0.5:
            score += 25
        elif mid_ela_ratio > 0.3:
            score += 15
        
        # Very low mean ELA can indicate the image was never JPEG-compressed 
        # (typical of AI-generated PNGs converted to JPEG via WhatsApp)
        if mean_ela < 3:
            score += 20
        elif mean_ela < 8:
            score += 10
        
        # Very high mean ELA = heavily re-compressed or manipulated
        if mean_ela > 40:
            score += 15
        
        score = min(100, max(0, score))
        
        return {
            "ela_score": round(score, 1),
            "mean_ela": round(mean_ela, 2),
            "std_ela": round(std_ela, 2),
            "uniformity": round(uniformity, 3),
            "low_ela_ratio": round(low_ela_ratio, 3),
            "mid_ela_ratio": round(mid_ela_ratio, 3),
        }
        
    except Exception as e:
        logger.warning("ELA analysis failed: %s", e)
        return {"ela_score": 50.0, "error": str(e)}


def analyze_frequency(image_path: str) -> dict:
    """Frequency domain analysis using FFT.
    
    AI-generated images typically have:
    - Less high-frequency noise (cameras add sensor noise)
    - More regular frequency patterns (GAN artifacts)
    - Different energy distribution in frequency space
    
    Returns dict with frequency_score (0-100, higher = more suspicious)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            img = np.array(Image.open(image_path).convert("RGB"))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Resize for consistent analysis
        gray = cv2.resize(gray, (256, 256))
        
        # 2D FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))
        
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Define frequency bands
        # Low: center 20%, Mid: 20-50%, High: 50-100%
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        max_dist = np.sqrt(center_y**2 + center_x**2)
        
        low_mask = dist < max_dist * 0.2
        mid_mask = (dist >= max_dist * 0.2) & (dist < max_dist * 0.5)
        high_mask = dist >= max_dist * 0.5
        
        low_energy = float(np.mean(magnitude[low_mask]))
        mid_energy = float(np.mean(magnitude[mid_mask]))
        high_energy = float(np.mean(magnitude[high_mask]))
        total_energy = low_energy + mid_energy + high_energy
        
        # Ratios
        if total_energy > 0:
            high_ratio = high_energy / total_energy
            low_ratio = low_energy / total_energy
        else:
            high_ratio = 0.33
            low_ratio = 0.33
        
        # Laplacian variance (measure of sharpness / high-freq content)
        gray_uint8 = gray.astype(np.uint8)
        laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
        lap_var = float(np.var(laplacian))
        
        # Noise estimation using median filter
        median_filtered = cv2.medianBlur(gray_uint8, 3).astype(np.float32)
        noise = gray - median_filtered
        noise_level = float(np.std(noise))
        
        # Spectral flatness (measure of how 'flat' the spectrum is)
        # AI images tend to have less flat spectra (more concentrated energy)
        flat_mag = magnitude.flatten()
        flat_mag = flat_mag[flat_mag > 0]
        if len(flat_mag) > 0:
            geometric_mean = np.exp(np.mean(np.log(flat_mag + 1e-10)))
            arithmetic_mean = np.mean(flat_mag)
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        else:
            spectral_flatness = 1.0
        
        # Compute suspicion score
        score = 0.0
        
        # AI images tend to have LESS high-frequency content
        if high_ratio < 0.20:
            score += 30
        elif high_ratio < 0.25:
            score += 20
        elif high_ratio < 0.28:
            score += 10
        
        # Very low noise level = suspicious (cameras always add some noise)
        if noise_level < 3.0:
            score += 25
        elif noise_level < 5.0:
            score += 15
        elif noise_level < 8.0:
            score += 5
        
        # Spectral flatness: lower = more structured = potentially AI
        if spectral_flatness < 0.3:
            score += 15
        elif spectral_flatness < 0.5:
            score += 10
        
        # Very low Laplacian variance = too smooth for a real photo
        if lap_var < 100:
            score += 20
        elif lap_var < 500:
            score += 10
        
        # Very high Laplacian variance can indicate sharpening artifacts
        if lap_var > 5000:
            score += 10
        
        score = min(100, max(0, score))
        
        return {
            "frequency_score": round(score, 1),
            "high_freq_ratio": round(high_ratio, 4),
            "low_freq_ratio": round(low_ratio, 4),
            "laplacian_variance": round(lap_var, 1),
            "noise_level": round(noise_level, 2),
            "low_energy": round(low_energy, 2),
            "mid_energy": round(mid_energy, 2),
            "high_energy": round(high_energy, 2),
        }
        
    except Exception as e:
        logger.warning("Frequency analysis failed: %s", e)
        return {"frequency_score": 50.0, "error": str(e)}


def compute_forensic_score(image_path: str) -> dict:
    """Combined forensic analysis score.
    
    Combines ELA + Frequency analysis into a single 0-100 suspicion score.
    This replaces the placeholder artifact detection with real forensics.
    """
    ela = analyze_ela(image_path)
    freq = analyze_frequency(image_path)
    
    ela_score = ela.get("ela_score", 50)
    freq_score = freq.get("frequency_score", 50)
    
    # Weighted combination: ELA 55%, Frequency 45%
    combined = 0.55 * ela_score + 0.45 * freq_score
    
    return {
        "forensic_score": round(combined, 1),
        "ela": ela,
        "frequency": freq,
    }
