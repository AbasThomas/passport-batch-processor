"""Image quality assessment — reports limitations instead of hiding them."""
import cv2
import numpy as np


def analyze(bgr, min_side=300):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(gray.mean())
    dark = float((gray < 40).mean())
    bright = float((gray > 235).mean())

    warnings = []
    if min(h, w) < min_side:
        warnings.append(f'Low resolution ({w}x{h}); output may look soft')
    if blur < 60:
        warnings.append('Image is blurry; fine facial detail may be lost')
    if mean < 70:
        warnings.append('Image is underexposed')
    elif mean > 200:
        warnings.append('Image is overexposed')
    if dark > 0.35:
        warnings.append('Large dark areas detected')
    if bright > 0.45:
        warnings.append('Large blown-out areas detected')

    grade = 'good'
    if blur < 60 or min(h, w) < min_side or mean < 60 or mean > 210:
        grade = 'poor'
    elif blur < 150 or min(h, w) < 500:
        grade = 'fair'

    return {
        'grade': grade,
        'width': w, 'height': h,
        'blur_score': round(blur, 1),
        'mean_luma': round(mean, 1),
        'warnings': warnings,
    }
