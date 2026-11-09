"""BiRefNet portrait/general provider (via rembg session 'birefnet-general')."""
from rembg import remove

from .segmentation_service import get_session

NAME = 'birefnet-general'


def segment(pil_image):
    return remove(pil_image, session=get_session(NAME))
