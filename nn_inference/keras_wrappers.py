import keras
import numpy as np
from typing import Union, Any
from .base_wrapper import BaseWrapper, Image, Descriptor


class KerasBase(BaseWrapper):
    def __init__(self, config: Union[str, Any], model: keras.models.Model):
        super().__init__()
        self.model = model
        self.config = config
        if isinstance(model, (str)):
            self.config = self.load_config(config)
        else:
            self.config = config

    def preprocess_image(self, image: Image) -> Image:
        return image / 255.0

    def predict(self, data: Image) -> Descriptor:
        descriptor = self.model.predict(data[np.newaxis, ...])
        return descriptor
