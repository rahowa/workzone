import cv2
import numpy as np
from app.base_types import Image


def encode_image(image: Image) -> bytes:
    encoded_img = cv2.imencode(image, 'jpeg')[1]
    return encoded_img.tobytes()

def deocode_image(encoded_image: bytes) -> Image:
    image = np.fromstring(encoded_image, dtype=np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)
