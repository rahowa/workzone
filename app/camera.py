import cv2
import numpy as np
from typing import List, Any, Union
Image = np.ndarray


class CamReader:
    """
    """
    def __init__(self, cam_id: int, height: int=480, width: int=640):
        self.cam_id = cam_id
        self.height = height
        self.width = width 

        self.capture = cv2.VideoCapture(cam_id)
        self.capture.set(3, width)
        self.capture.set(4, height)

    def __repr__(self):
        return f"CamReader_{self.cam_id}"

    def __str__(self):
        return f"CamReader_{self.cam_id}"

    def get_frame(self, enchance: Union[Any, List[Any]]=None) -> Image:
        _, frame = self.capture.read()

        if enchance is not None:
            if isinstance(enchance, (list, tuple)):
                for ench in enchance:
                    frame = ench(frame)
            # elif isinstance(enchance, callable):
            else:
                frame = enchance(frame)
        return frame

    def gen_frame(self) -> Image:
        while True:
            yield self.get_frame()

    def gen_encoded_frame(self, enchance: Union[Any, List[Any]]=None) -> bytes:
        while True:
            frame = self.get_frame(enchance)
            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def __del__(self):
        if self.capture:
            self.capture.release()




def encode_image(image: Image) -> bytes:
    encoded_img = cv2.imencode(image, 'jpeg')[1]
    return encoded_img.tobytes()


def wrap_image(image: bytes) -> str:
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')