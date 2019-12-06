import json
import numpy as np
from typing import Dict, Any, List, Tuple
from abc import ABC, abstractmethod

Image = np.ndarray
Descriptor = np.ndarray
Descriptors = List[Descriptor]
BBox = Tuple[int, int, int, int]
BBoxes = Tuple[BBox, ...]


class BaseWrapper(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def preprocess_image(self, *args, **kwargs):
        raise NotImplementedError

    def load_config(self, path_to_config: str) -> Dict[str, Any]:
        with open(path_to_config, 'r') as conf_file:
            config = json.load(conf_file)
        return config
