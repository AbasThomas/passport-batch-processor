"""Validation of an extracted photograph before downstream processing."""
import cv2
import numpy as np

MIN_SIDE = 120          # px; below this the print is unusable
ASPECT_RANGE = (0.4, 2.4)


def validate_extraction(bgr):
    """Return (ok, warnings, metrics) for a rectified photo."""
    h, w = bgr.shape[:2]
    warnings, metrics = [], {}
    metrics['width'], metrics['height'] = w, h
    metrics['aspect'] = round(w / max(h, 1), 3)

    if min(w, h) < MIN_SIDE:
        warnings.append(f'Extracted photo is very small ({w}x{h}px); quality will be poor')
    if not (ASPECT_RANGE[0] <= metrics['aspect'] <= ASPECT_RANGE[1]):
        warnings.append(f'Unusual photo aspect ratio ({metrics["aspect"]})')

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # blur: Laplacian variance
    metrics['blur_score'] = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)
    if metrics['blur_score'] < 40:
        warnings.append('Extracted photo appears blurry')
    # border cleanliness: outer 3% should not be near-uniform paper white everywhere
    # (a fully blank border often means we captured margin, not the photo)
    border = np.concatenate([
        gray[: max(2, int(h * 0.03))].ravel(),
        gray[-max(2, int(h * 0.03)):].ravel(),
        gray[:, : max(2, int(w * 0.03))].ravel(),
        gray[:, -max(2, int(w * 0.03)):].ravel(),
    ])
    metrics['border_std'] = round(float(border.std()), 1)
    if border.std() < 4 and gray.std() > 25:
        warnings.append('Photo borders look like blank paper; crop may include margin')

    metrics['content_std'] = round(float(gray.std()), 1)
    if gray.std() < 8:
        warnings.append('Extracted region has almost no content')

    ok = min(w, h) >= MIN_SIDE and gray.std() >= 8
    return ok, warnings, metrics
