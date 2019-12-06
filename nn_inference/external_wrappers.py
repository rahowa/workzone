import numpy as np
import face_recognition as fr
from typing import Union, Any
from .base_wrapper import BaseWrapper, Image, Descriptor, Descriptors


class FaceDetectionLibWrapper(BaseWrapper):
    def __init__(self, config: Union[str, Any], model: keras.models.Model):
        super().__init__()
        self.config = config
        if isinstance(config, (str)):
            self.config = self.load_config(config)
        else:
            self.config = config
        self.model = config['model_type']
        self.n_upsample = config['number_of_times_to_upsample']

    def preprocess_image(self, image: Image) -> Image:
        return image / 255.0

    def predict(self, data: Image) -> Descriptors:
        data = self.preprocess_image(data)
        face_locations = fr.face_locations(data, 
                                           number_of_times_to_upsample=self.n_upsample,
                                           model=self.model)
        face_encodings = fr.face_encodings(data, face_locations)
        return face_encodings
