"""U2Net human-segmentation fallback provider."""
from rembg import remove

from .segmentation_service import get_session

NAME = 'u2net'


def segment(pil_image):
    return remove(pil_image, session=get_session(NAME))
