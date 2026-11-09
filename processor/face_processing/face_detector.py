"""Face detection via OpenCV YuNet (5-point landmarks included).

Returns a list of detections; callers pick the intended face when several appear.
"""
import cv2
import numpy as np

from .yunet_model import ensure_model

_detector = None
_detector_size = None


def _get_detector(input_size=(320, 320)):
    global _detector, _detector_size
    # YuNet requires fixed input size; recreate when it changes materially.
    if _detector is None or _detector_size != input_size:
        _detector = cv2.FaceDetectorYN.create(
            ensure_model(), '', input_size,
            score_threshold=0.6, nms_threshold=0.3, top_k=32)
        _detector_size = input_size
    return _detector


def detect_faces(bgr, score_threshold=0.6):
    """Detect faces. Returns list of dicts sorted by confidence desc:
    {bbox:[x,y,w,h], landmarks:[[x,y]x5], score, face_index}.
    Landmarks order (YuNet): right eye, left eye, nose, right mouth, left mouth.
    """
    h, w = bgr.shape[:2]
    # run at native resolution capped for speed
    scale = min(1.0, 1600 / max(h, w))
    img = cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale < 1 else bgr
    ih, iw = img.shape[:2]
    det = _get_detector((iw, ih))
    det.setInputSize((iw, ih))
    _, faces = det.detect(img)
    out = []
    if faces is not None:
        for i, f in enumerate(faces):
            score = float(f[14])
            if score < score_threshold:
                continue
            bbox = (f[0:4] / scale).astype(float).tolist()
            lm = (f[4:14].reshape(5, 2) / scale).astype(float).tolist()
            out.append({'bbox': bbox, 'landmarks': lm, 'score': round(score, 4),
                        'face_index': i})
    out.sort(key=lambda d: d['score'], reverse=True)
    return out


def best_face(bgr, preferred_index=None):
    """Return (face_dict, all_faces). preferred_index selects among multiple."""
    faces = detect_faces(bgr)
    if not faces:
        return None, []
    if preferred_index is not None and 0 <= preferred_index < len(faces):
        return faces[preferred_index], faces
    return faces[0], faces
