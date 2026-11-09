"""Composite the segmented person onto a uniform replacement background."""
import numpy as np


def parse_color(value):
    """'#rrggbb' or common names -> (r,g,b) tuple."""
    named = {
        'white': (255, 255, 255),
        'lightgrey': (240, 240, 240), 'lightgray': (240, 240, 240),
        'lightblue': (221, 235, 247),
    }
    if not value:
        return (255, 255, 255)
    v = str(value).strip().lower()
    if v in named:
        return named[v]
    if v.startswith('#') and len(v) == 7:
        return tuple(int(v[i:i + 2], 16) for i in (1, 3, 5))
    if v.startswith('#') and len(v) == 4:  # #rgb shorthand
        return tuple(int(c * 2, 16) for c in v[1:4])
    if len(v) == 6:
        return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))
    return (255, 255, 255)


def composite(rgb, alpha, color):
    """rgb uint8 HxWx3, alpha uint8 HxW -> composited uint8 HxWx3."""
    fg = rgb.astype(np.float32)
    bg = np.empty_like(fg)
    bg[:] = color
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    return np.clip(fg * a + bg * (1 - a), 0, 255).astype(np.uint8)
