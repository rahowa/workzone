import cv2
import numpy as np
import base64

from app.base_types import Image


def encode_image(image: Image) -> bytes:
    encoded_img = cv2.imencode('jpeg', image)[1]
    return encoded_img.tobytes()


def image_to_str(image: Image) -> str:
    encoded_img = cv2.imencode(".jpeg", image)[1]
    return base64.encodebytes(encoded_img.tobytes()).decode("utf-8")


def deocode_image(encoded_image: bytes) -> Image:
    image = np.fromstring(encoded_image, dtype=np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)
