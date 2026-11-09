"""Manual four-corner crop support for low-confidence automatic detection."""
import numpy as np

from .perspective_corrector import rectify


def normalize_corners(corners, img_w, img_h):
    """Accept corners as normalized [0..1] or pixel coords -> pixel float array."""
    arr = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    if arr.max() <= 1.5 and arr.min() >= -0.01:
        arr[:, 0] *= img_w
        arr[:, 1] *= img_h
    return arr


def manual_rectify(bgr, corners):
    """Rectify using user-supplied corners. Raises ValueError on bad input."""
    h, w = bgr.shape[:2]
    pts = normalize_corners(corners, w, h)
    if pts[:, 0].min() < -w * 0.05 or pts[:, 0].max() > w * 1.05 \
            or pts[:, 1].min() < -h * 0.05 or pts[:, 1].max() > h * 1.05:
        raise ValueError('corners outside image bounds')
    warped, ow, oh = rectify(bgr, pts.tolist())
    if min(ow, oh) < 40:
        raise ValueError('selected region too small')
    return warped, {'width': ow, 'height': oh, 'manual': True}
