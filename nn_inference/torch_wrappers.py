import torch 
import numpy as np 
from typing import Union, Any

from .base_wrapper import BaseWrapper, Image, Descriptor


class PytorchBase(BaseWrapper):
    def __init__(self, config: Union[str, Any], model: torch.Module):
        super().__init__()
        self.model = model
        self.config = config
        if isinstance(model, (str)):
            self.config = self.load_config(config)
        else:
            self.config = config

    def preprocess_image(self, image: Image) -> Image:
        device = self.config['device']
        image = torch.from_numpy(image).to(device)
        image = image.unsqueeze(0)
        image /= 255.0
        return image

    def predict(self, data: Image) -> Descriptor:
        data = self.preprocess_image(data)
        self.model.eval()
        with torch.no_grad():
            descriptor = self.model(data)
            descriptor = descriptor.cpu().numpy()
        return descriptor