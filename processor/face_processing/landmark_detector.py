"""Facial geometry derived from YuNet 5-point landmarks.

Landmark order: right eye, left eye, nose tip, right mouth corner, left mouth corner
(image coordinates; 'right'/'left' are the person's right/left, i.e. mirrored in image).
"""
import numpy as np


def eye_centers(landmarks):
    lm = np.asarray(landmarks, dtype=np.float32)
    # lm[0]=person-right eye (image-left), lm[1]=person-left eye (image-right)
    return lm[0], lm[1]


def eye_angle_deg(landmarks):
    """Angle of the inter-eye line; positive means the person's right eye is lower."""
    r, l = eye_centers(landmarks)
    return float(np.degrees(np.arctan2(l[1] - r[1], l[0] - r[0])))


def face_geometry(face, img_shape):
    """Compute crop-relevant measurements from one detection."""
    lm = np.asarray(face['landmarks'], dtype=np.float32)
    x, y, w, h = face['bbox']
    r_eye, l_eye = lm[0], lm[1]
    nose, r_mouth, l_mouth = lm[2], lm[3], lm[4]

    eye_mid = (r_eye + l_eye) / 2
    mouth_mid = (r_mouth + l_mouth) / 2
    eyes_dist = float(np.linalg.norm(l_eye - r_eye))
    face_w = float(w)
    face_h = float(h)

    # --- head pose proxies ---
    # Inter-eye distance relative to bbox width: frontal faces ~0.30-0.45;
    # strong profiles compress it below ~0.24.
    eye_ratio = eyes_dist / max(face_w, 1)
    # Eye midpoint offset from bbox horizontal center: yaw proxy.
    eye_offset = abs(eye_mid[0] - (x + w / 2)) / max(w, 1)

    yaw_score = min(eye_ratio / 0.32, 1.0) * max(0.0, 1.0 - eye_offset / 0.30)
    frontal = yaw_score >= 0.5 and eye_ratio >= 0.24

    # chin estimate: mouth corners sit ~1/3 of the way from eyes to chin
    eye_to_mouth = float(np.linalg.norm(mouth_mid - eye_mid))
    chin_y = float(mouth_mid[1] + eye_to_mouth * 1.15)
    chin_x = float(mouth_mid[0] + (mouth_mid[0] - eye_mid[0]) * 0.4)
    # crown estimate: forehead extends roughly one eye-to-chin distance above eyes
    eye_to_chin = max(chin_y - eye_mid[1], 1.0)
    crown_y = float(eye_mid[1] - eye_to_chin * 0.95)

    return {
        'bbox': [float(x), float(y), float(w), float(h)],
        'eyes': {'right': r_eye.tolist(), 'left': l_eye.tolist(),
                 'mid': eye_mid.tolist(), 'distance': eyes_dist},
        'nose': nose.tolist(), 'mouth_mid': mouth_mid.tolist(),
        'eye_angle_deg': eye_angle_deg(face['landmarks']),
        'chin': [chin_x, chin_y],
        'crown_y': crown_y,
        'eye_to_chin': float(eye_to_chin),
        'head_height': float(chin_y - crown_y),
        'eye_ratio': round(float(eye_ratio), 3),
        'yaw_score': round(float(yaw_score), 3),
        'frontal': bool(frontal),
        'score': face.get('score', 0.0),
    }
