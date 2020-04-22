import cv2
from flask import g
import numpy as np
from typing import List, Any, Union, Iterator, Callable

from app.base_types import Image


class CamReader:
    """
    """
    def __init__(self, cam_id: int, height: int = 480, width: int = 640):
        self.cam_id = cam_id
        self.height = height
        self.width = width 

        self.capture = cv2.VideoCapture(cam_id)
        self.capture.set(3, width)
        self.capture.set(4, height)

    def __repr__(self):
        return f"CamReader_{self.cam_id}"

    def get_frame(self,
                  enchance: Union[Callable[[Image], Image], List[Callable[[Image], Image]]] = None) -> Image:
        _, frame = self.capture.read()

        if enchance is not None:
            if isinstance(enchance, (list, tuple)):
                for ench in enchance:
                    frame = ench(frame)
            else:
                frame = enchance(frame)
        return frame

    def gen_frame(self) -> Image:
        while True:
            yield self.get_frame()

    def gen_encoded_frame(self, enchance: Union[Any, List[Any]] = None) -> Iterator[bytes]:
        while True:
            frame = self.get_frame(enchance)
            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

    def __del__(self):
        if self.capture:
            self.capture.release()


def get_video_capture(cam_id: int) -> CamReader:
    caputure = getattr(g, "_capture", None)
    if caputure is None:
        caputure = g._capture = CamReader(cam_id)
    return caputure