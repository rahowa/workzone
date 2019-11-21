import cv2
import numpy as np

Image = np.ndarray


class CamReader:
    """
    """

    def __init__(self, cam_id: int):
        self.cam_id = cam_id
        self.capture = cv2.VideoCapture(cam_id)

    def __repr__(self):
        return f"CamReader_{self.cam_id}"

    def __str__(self):
        return f"CamReader_{self.cam_id}"

    def get_frame(self) -> Image:
        _, frame = self.capture.read()
        return frame

    def gen_frame(self) -> Image:
        while True:
            yield self.get_frame()

    def gen_encoded_frame(self) -> bytes:
        while True:
            frame = self.get_frame()
            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def __del__(self):
        if self.capture:
            self.capture.release()
