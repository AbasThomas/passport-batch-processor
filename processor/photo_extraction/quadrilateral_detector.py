"""Quadrilateral geometry helpers shared by detection strategies."""
import cv2
import numpy as np


def order_points(pts):
    """Order 4 points as tl, tr, br, bl (works for mildly rotated quads)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)], pts[np.argmin(d)],
        pts[np.argmax(s)], pts[np.argmax(d)],
    ], dtype=np.float32)


def quad_area(pts):
    p = order_points(pts)
    return 0.5 * abs(
        np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))
    )


def quad_angles(pts):
    p = order_points(pts)
    out = []
    for i in range(4):
        a = p[(i - 1) % 4]
        b = p[i]
        c = p[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        out.append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
    return out


def is_valid_quad(pts, img_shape, min_area_frac=0.02, max_area_frac=0.95,
                  min_angle=35.0, max_angle=145.0):
    h, w = img_shape[:2]
    area = quad_area(pts)
    frac = area / float(h * w)
    if not (min_area_frac <= frac <= max_area_frac):
        return False, 'area_out_of_range'
    angles = quad_angles(pts)
    if min(angles) < min_angle or max(angles) > max_angle:
        return False, 'not_rectangular'
    p = order_points(pts)
    if (p < -0.05 * max(h, w)).any() or (p[:, 0] > w * 1.05).any() or (p[:, 1] > h * 1.05).any():
        return False, 'outside_image'
    return True, 'ok'


def quad_aspect(pts):
    """Width/height aspect of the quad using ordered edge lengths."""
    p = order_points(pts)
    top = np.linalg.norm(p[1] - p[0])
    bottom = np.linalg.norm(p[2] - p[3])
    left = np.linalg.norm(p[3] - p[0])
    right = np.linalg.norm(p[2] - p[1])
    wlen = (top + bottom) / 2
    hlen = (left + right) / 2
    return (wlen / hlen) if hlen > 1 else 0.0
