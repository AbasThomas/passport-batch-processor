"""Head-roll correction: rotate the image so the eyes are horizontal.

Only minor roll is corrected automatically; large angles are reported so the
caller can ask for a better source rather than distorting the face.
"""
import cv2
import numpy as np

MAX_AUTO_ROLL_DEG = 15.0


def roll_angle(geom):
    return float(geom.get('eye_angle_deg', 0.0))


def deskew(bgr, geom):
    """Rotate image to level the eyes.

    Returns (image, applied_deg, status) where status is
    'aligned' | 'skipped_small' | 'too_large'.
    """
    angle = roll_angle(geom)
    if abs(angle) < 0.4:
        return bgr, 0.0, 'skipped_small'
    if abs(angle) > MAX_AUTO_ROLL_DEG:
        return bgr, angle, 'too_large'
    h, w = bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    # positive angle -> rotate clockwise to bring right-eye-lower line level
    rotated = cv2.warpAffine(
        bgr, m, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle, 'aligned'
