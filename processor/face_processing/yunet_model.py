"""YuNet face-detection model download/cache.

OpenCV ships cv2.FaceDetectorYN; only the ONNX weights need to be fetched once.
Override location with YUNET_MODEL_PATH, or set MODEL_DIR for the cache dir.
"""
import os
import urllib.request
from pathlib import Path

MODEL_URL = ('https://github.com/opencv/opencv_zoo/raw/main/models/'
             'face_detection_yunet/face_detection_yunet_2023mar.onnx')
MODEL_NAME = 'face_detection_yunet_2023mar.onnx'


def model_path():
    override = os.getenv('YUNET_MODEL_PATH')
    if override:
        return Path(override)
    model_dir = Path(os.getenv('MODEL_DIR', Path(__file__).resolve().parents[1] / 'models'))
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / MODEL_NAME


def ensure_model():
    p = model_path()
    if p.exists() and p.stat().st_size > 100_000:
        return str(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.onnx.part')
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.replace(p)
    return str(p)
