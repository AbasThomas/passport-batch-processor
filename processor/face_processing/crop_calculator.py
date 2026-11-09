"""Automatic passport-crop engine driven by document profiles.

A profile defines target aspect ratio and what fraction of frame height the
head (crown->chin) should occupy. The crop is computed from facial geometry;
no face stretching or reconstruction is ever applied.
"""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class CropProfile:
    name: str
    aspect: float          # output width / height
    head_fraction: float   # crown->chin occupies this fraction of frame height
    min_head_fraction: float = 0.60
    max_head_fraction: float = 0.85
    shoulder_fraction: float = 2.6  # frame width as multiple of face width


PROFILES = {
    # ICAO-style 35x45mm @300dpi -> 413x531 (~0.78)
    'icao35x45': CropProfile('icao35x45', 0.778, 0.72, 0.65, 0.80, 2.4),
    # US passport/visa 2x2in @300dpi -> 600x600
    'us2x2': CropProfile('us2x2', 1.0, 0.50, 0.45, 0.60, 2.2),
    # Common 35x45 alternative square-ish
    'square': CropProfile('square', 1.0, 0.55, 0.50, 0.65, 2.3),
}

DEFAULT_PROFILE = 'icao35x45'


def get_profile(name):
    return PROFILES.get(name or DEFAULT_PROFILE, PROFILES[DEFAULT_PROFILE])


def compute_crop(geom, img_w, img_h, profile, zoom=1.0, pan_x=0.0, pan_y=0.0):
    """Return (x, y, w, h) pixel crop centered on the face.

    zoom>1 crops tighter; pan_x/pan_y shift the crop in fractions of its size.
    The crop is clamped inside the image; if the image is too small the largest
    valid centered crop is returned and 'fits' is False.
    """
    p = profile
    face_w = geom['bbox'][2]
    head_h = max(geom['head_height'], 1.0)

    # frame height so that head occupies head_fraction (zoom tightens it)
    frame_h = head_h / (p.head_fraction * zoom)
    frame_w = frame_h * p.aspect

    # ensure shoulders are plausibly included; widen if needed
    min_w = face_w * p.shoulder_fraction
    if frame_w < min_w:
        frame_w = min_w
        frame_h = frame_w / p.aspect

    # anchor: crown at (1-head_fraction)/2 of frame from top, horizontally centered
    crown_y = geom['crown_y']
    eye_mid_x = geom['eyes']['mid'][0]
    chin_x = geom['chin'][0]
    center_x = (eye_mid_x + chin_x) / 2.0

    x = center_x - frame_w / 2.0 + pan_x * frame_w
    y = crown_y - frame_h * (1.0 - p.head_fraction) / 2.0 - head_h * 0.02 + pan_y * frame_h

    # clamp into image
    x = min(max(x, 0.0), max(0.0, img_w - frame_w))
    y = min(max(y, 0.0), max(0.0, img_h - frame_h))
    frame_w = min(frame_w, float(img_w))
    frame_h = min(frame_h, float(img_h))

    fits = frame_w >= img_w * 0.95 and frame_h >= img_h * 0.95
    return (int(round(x)), int(round(y)), int(round(frame_w)), int(round(frame_h))), fits


def output_size(profile, dpi=300):
    """Pixel dimensions for common print sizes at given DPI."""
    mm = {
        'icao35x45': (35, 45),
        'us2x2': (50.8, 50.8),
        'square': (35, 35),
    }
    w_mm, h_mm = mm.get(profile.name, (35, 45))
    return (round(w_mm / 25.4 * dpi), round(h_mm / 25.4 * dpi))
