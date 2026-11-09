"""Perspective rectification of a detected photo quadrilateral."""
import cv2
import numpy as np

from .quadrilateral_detector import order_points


def rectify(bgr, points, max_output_dim=4000):
    """Warp a quad region to a rectangle. Returns (warped_bgr, out_w, out_h)."""
    p = order_points(points).astype(np.float32)
    tl, tr, br, bl = p
    ow = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    oh = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    if ow < 8 or oh < 8:
        raise ValueError('quad too small')
    scale = min(1.0, max_output_dim / max(ow, oh))
    ow, oh = max(8, int(ow * scale)), max(8, int(oh * scale))
    dst = np.float32([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]])
    m = cv2.getPerspectiveTransform(p, dst)
    return cv2.warpPerspective(bgr, m, (ow, oh)), ow, oh
