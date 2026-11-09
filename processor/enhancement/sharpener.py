"""Mild unsharp masking. Never generative — no detail is invented."""
import cv2


def sharpen(bgr, amount=0.6, radius=1.0):
    blur = cv2.GaussianBlur(bgr, (0, 0), radius)
    diff = cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)
    return diff


def maybe_sharpen(bgr, enabled=True):
    if not enabled:
        return bgr, {'applied': False}
    return sharpen(bgr), {'applied': True, 'amount': 0.6}
