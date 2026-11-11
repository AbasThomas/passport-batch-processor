"""Person-segmentation provider abstraction.

Providers wrap rembg ONNX sessions so models can be swapped without touching
the pipeline. All inference is local (ONNX Runtime); nothing leaves the host.
"""
from rembg import new_session, remove

_sessions = {}

# preferred order when 'auto' is requested — smallest first for faster startup
AUTO_ORDER = ('u2net', 'isnet-general-use', 'birefnet-general')


def get_session(model_name):
    if model_name not in _sessions:
        _sessions[model_name] = new_session(model_name)
    return _sessions[model_name]


def available_models():
    return list(AUTO_ORDER)


def segment(pil_image, model_name='auto'):
    """Return RGBA PIL image with alpha matte. Falls back through AUTO_ORDER."""
    names = AUTO_ORDER if model_name in (None, 'auto', '') else (model_name,)
    last_err = None
    for name in names:
        try:
            return remove(pil_image, session=get_session(name)), name
        except Exception as e:  # model missing weights etc -> try next
            last_err = e
    raise RuntimeError(f'All segmentation providers failed: {last_err}')


def alpha_stats(rgba_array):
    import numpy as np
    a = rgba_array[:, :, 3]
    coverage = float((a > 18).mean())
    return {'coverage': round(coverage, 4),
            'mean_alpha': round(float(a.mean()), 1)}
