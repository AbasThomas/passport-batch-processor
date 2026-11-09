"""Face alignment stage: detect -> landmarks -> geometry -> deskew."""
import cv2
import numpy as np

from .face_detector import best_face
from .landmark_detector import face_geometry
from . import face_aligner


def align(bgr, preferred_index=None):
    """Run detection + geometry + deskew.

    Returns dict with keys: ok, status, reason, faces, geom, image, roll_deg.
    status: 'ok' | 'no_face' | 'multiple_faces' | 'profile' | 'roll_too_large' | 'low_confidence'
    """
    face, faces = best_face(bgr, preferred_index)
    if face is None:
        return {'ok': False, 'status': 'no_face', 'faces': [],
                'reason': 'No face detected; please upload a clearer photograph',
                'image': bgr}
    if len(faces) > 1 and preferred_index is None:
        return {'ok': False, 'status': 'multiple_faces', 'faces': faces,
                'reason': 'Multiple faces detected; select the intended person',
                'image': bgr}
    if face['score'] < 0.75:
        return {'ok': False, 'status': 'low_confidence', 'faces': faces,
                'reason': 'Face detection confidence is low',
                'image': bgr}

    geom = face_geometry(face, bgr.shape)
    if not geom['frontal']:
        return {'ok': False, 'status': 'profile', 'faces': faces, 'geom': geom,
                'reason': 'Face appears turned sideways; a more frontal photograph is required',
                'image': bgr}

    image, roll, roll_status = face_aligner.deskew(bgr, geom)
    if roll_status == 'too_large':
        return {'ok': False, 'status': 'roll_too_large', 'faces': faces, 'geom': geom,
                'reason': f'Head tilt of {roll:.1f}° is too large to correct safely',
                'image': bgr}
    if roll_status == 'aligned':
        geom = _rotate_geometry(geom, roll, bgr.shape)
    return {'ok': True, 'status': 'ok', 'faces': faces, 'geom': geom,
            'image': image, 'roll_deg': roll, 'roll_status': roll_status}


def _rotate_geometry(geom, angle_deg, shape):
    """Apply the same rotation deskew used to all geometry coordinates."""
    import copy
    h, w = shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)

    def xf(pt):
        v = m @ np.array([pt[0], pt[1], 1.0])
        return [float(v[0]), float(v[1])]

    g = copy.deepcopy(geom)
    x, y, bw, bh = g['bbox']
    g['bbox'] = [x, y, bw, bh]  # bbox rotates with image; refined below
    g['eyes']['right'] = xf(g['eyes']['right'])
    g['eyes']['left'] = xf(g['eyes']['left'])
    g['eyes']['mid'] = xf(g['eyes']['mid'])
    g['nose'] = xf(g['nose'])
    g['mouth_mid'] = xf(g['mouth_mid'])
    g['chin'] = xf(g['chin'])
    g['crown_y'] = xf([g['eyes']['mid'][0], g['crown_y']])[1]
    # recompute axis-aligned bbox around rotated eyes+chin approx
    xs = [g['eyes']['mid'][0], g['chin'][0]]
    ys = [g['eyes']['mid'][1], g['chin'][1], g['crown_y']]
    g['bbox'] = [min(xs) - bw * 0.25, min(ys) - bh * 0.1,
                 bw * 1.5, max(ys) - min(ys) + bh * 0.2]
    return g
