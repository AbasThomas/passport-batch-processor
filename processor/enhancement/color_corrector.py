"""White balance + exposure + contrast correction. Conservative by design."""
import cv2
import numpy as np


def gray_world_white_balance(bgr, strength=0.5):
    """Scale channels so their means match the green channel mean.

    strength<1 blends toward the original to preserve skin tones.
    """
    img = bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    if means.min() < 5:
        return bgr
    target = means[1]  # green as reference
    gains = np.clip(target / np.maximum(means, 1.0), 0.85, 1.18)
    gains = 1.0 + (gains - 1.0) * strength
    out = img * gains[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def correct_exposure(bgr):
    """If mean luminance is far from mid-grey, apply a gentle gamma/exposure fix."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    if 95 <= mean <= 175:
        return bgr, {'applied': False, 'mean_luma': round(mean, 1)}
    # target ~135; gamma = log(0.5)/log(mean/255)
    gamma = float(np.clip(np.log(0.5) / np.log(max(mean, 1) / 255.0), 0.75, 1.4))
    lut = np.array([min(255, int(255 * (i / 255.0) ** gamma)) for i in range(256)],
                   dtype=np.uint8)
    return cv2.LUT(bgr, lut), {'applied': True, 'mean_luma': round(mean, 1),
                               'gamma': round(gamma, 3)}


def correct_contrast(bgr, clip=2.0):
    """CLAHE on L channel of LAB — mild, preserves colour."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, ch]), cv2.COLOR_LAB2BGR)
