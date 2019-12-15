import numpy as np
import face_recognition as fr
from typing import Union, Any
from .base_wrapper import BaseWrapper, Image, Descriptor, Descriptors, BBoxes


class FaceDetectionLibWrapper(BaseWrapper):
    def __init__(self, config: Union[str, Any]):
        super().__init__()
        self.config = config
        if isinstance(config, (str)):
            self.config = self.load_config(config)
        else:
            self.config = config
        self.model = config['model_type']
        self.n_upsample = config['number_of_times_to_upsample']

    def preprocess_image(self, image: Image) -> Image:
        if image.mean() > 1.0:
            return image / 255.0
        else:
            return image

    def get_locations(self, data: Image) -> BBoxes:
        face_locations = fr.face_locations(data,
                                           number_of_times_to_upsample=self.n_upsample,
                                           model=self.model
        return face_locations

    def get_encodings(self, data: Image, bboxes: BBoxes) -> Descriptors:
        face_encodings = fr.face_encodings(data, bboxes)
        return face_encodings

    def predict(self, data: Image) -> Descriptors:
        data = self.preprocess_image(data)
        face_locations = self.get_locations(data)
        face_encodings = self.get_encodings(data, face_locations)
        return face_encodings
