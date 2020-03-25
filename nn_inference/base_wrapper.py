import json
import numpy as np
from abc import ABC, abstractmethod
from nptyping import Array
from typing import Dict, Any, List, Tuple

Image = Array[int]
Descriptor = Array[float]
Descriptors = Tuple[Array[float], ...]
BBox = Tuple[int, int, int, int]
BBoxes = Tuple[BBox, ...]


class BaseWrapper(ABC):
    """
    Base class for creating custom wrappers for 
    models based on neural networks
    """
    @abstractmethod
    def predict(self, image: Image):
        """
        Abstract method for predict result based on
        input image
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess_image(self, *args, **kwargs):
        """
        Abstract method for image preprocessing
        for certain model/framework
        """
        raise NotImplementedError

    def load_config(self, path_to_config: str) -> Dict[str, Any]:
        """
        Generic method for loading json config

        Parameters
        ----------
            path_to_config: str
                Path to config file

        Returns
        -------
            config: Dict[str, Any]
                Model config in dictionary
        """

        with open(path_to_config, 'r') as conf_file:
            config = json.load(conf_file)
        return config
