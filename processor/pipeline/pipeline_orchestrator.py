"""Service orchestrator: session warming, preview analysis, batch dispatch."""
import cv2
import numpy as np

from photo_extraction.boundary_detector import detect_photograph, best_candidate
from photo_extraction.perspective_corrector import rectify
from face_processing.face_detector import detect_faces
from face_processing.aligner import align
from .passport_pipeline import run as run_one, Review


def analyze(input_path):
    """Preview for the UI: extraction candidates + detected faces, no export."""
    src = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if src is None:
        raise Review('Unreadable image')
    h, w = src.shape[:2]

    candidates = detect_photograph(src)
    top, conf = best_candidate(candidates)
    preview = None
    if top is not None:
        photo, ow, oh = rectify(src, top['points'])
        small = cv2.resize(photo, (min(ow, 480), int(oh * min(ow, 480) / ow)))
        ok, enc = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        import base64
        preview = base64.b64encode(enc.tobytes()).decode() if ok else None

    # faces on the best-guess extraction — but if that region has no face while
    # the full frame does, the boundary is probably wrong; analyze the full frame
    work = src
    if top is not None and conf >= 0.3:
        scale = min(1.0, 480 / max(photo.shape[:2]))
        probe = cv2.resize(photo, (int(photo.shape[1] * scale), int(photo.shape[0] * scale)))
        if detect_faces(probe):
            work = photo
    faces = detect_faces(work)
    alignment = align(work)

    return {
        'width': w, 'height': h,
        'extraction': {
            'confidence': conf,
            'autoAccept': conf >= 0.45,
            'candidates': candidates[:5],
            'previewJpeg': preview,
        },
        'faces': [{'bbox': f['bbox'], 'score': f['score'],
                   'face_index': f['face_index']} for f in faces],
        'alignment': {'ok': alignment['ok'], 'status': alignment['status'],
                      'reason': alignment.get('reason')},
    }


def warmup():
    """Preload model sessions so the first request isn't slow."""
    import threading
    def _w():
        try:
            from face_processing.yunet_model import ensure_model
            from background_removal.segmentation_service import get_session
            ensure_model()
            get_session('u2net')  # smaller model for faster startup
        except Exception:
            pass
    threading.Thread(target=_w, daemon=True).start()


def process(input_path, output_path, settings):
    return run_one(input_path, output_path, settings)
