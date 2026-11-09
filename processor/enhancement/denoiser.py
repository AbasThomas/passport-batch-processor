"""Conservative denoising — only applied when noise is detected."""
import cv2
import numpy as np


def estimate_noise(gray):
    """MAD-based noise estimate on the Laplacian (Immerkær-style)."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.median(np.abs(lap))) / 0.6745


def denoise(bgr, strength=1):
    """FastNlMeansDenoisingColored; strength scales filter params mildly."""
    h = int(3 + strength)
    return cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)


def maybe_denoise(bgr, enabled=True):
    if not enabled:
        return bgr, {'applied': False}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    noise = estimate_noise(gray)
    if noise < 4.0:
        return bgr, {'applied': False, 'noise_estimate': round(noise, 2)}
    out = denoise(bgr, strength=1 if noise < 9 else 2)
    return out, {'applied': True, 'noise_estimate': round(noise, 2)}
