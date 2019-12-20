import keras
import numpy as np
from typing import Union, Any
from .base_wrapper import BaseWrapper, Image, Descriptors


class KerasBase(BaseWrapper):
    """
    Example wrapper for pure Keras face reсognition system
    """
    
    def __init__(self, config: Union[str, Any], model: keras.models.Model):
        super().__init__()
        self.model = model
        if isinstance(config, (str)):
            self.config = self.load_config(config)
        else:
            self.config = config

    def preprocess_image(self, image: Image) -> Image:
        return image / 255.0

    def predict(self, data: Image) -> Descriptors:
        descriptors = self.model.predict(data[np.newaxis, ...])
        return descriptors
