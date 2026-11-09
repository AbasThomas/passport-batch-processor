"""Mask refinement: morphology cleanup, feathering, halo suppression."""
import cv2
import numpy as np


def refine_alpha(alpha, feather_px=1.2):
    """alpha: uint8 HxW. Returns refined uint8 alpha."""
    a = alpha.copy()
    # remove speckle noise
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # close small holes inside the person
    a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # feather edges
    k = max(3, int(feather_px * 2) | 1)
    a = cv2.GaussianBlur(a, (k, k), 0)
    # re-threshold softly to keep hard interior, soft boundary
    a = cv2.GaussianBlur(a, (3, 3), 0)
    return a.astype(np.uint8)


def despill(rgb, alpha):
    """Reduce colour halo: pull semi-transparent edge pixels toward neutral."""
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    # where alpha is mid-range, desaturate slightly toward luminance
    lum = (0.299 * rgb[:, :, 2] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 0])
    edge = (alpha > 30) & (alpha < 220)
    out = rgb.astype(np.float32)
    for c in range(3):
        out[:, :, c] = np.where(edge, out[:, :, c] * 0.75 + lum * 0.25, out[:, :, c])
    return np.clip(out, 0, 255).astype(np.uint8)
