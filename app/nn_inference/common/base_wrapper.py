import json
from typing import Dict, Any, Sequence
from abc import ABC, abstractmethod

from typing import List
from app.base_types import Image
from app.result_types import BaseResult


class BaseWrapper(ABC):
    """
    Base class for creating custom wrappers for 
    models based on neural networks
    """
    @abstractmethod
    def predict(self, image: Image) -> List[BaseResult]:
        """
        Abstract method for predict result based on
        input image
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, image: Sequence[Image]) -> Any:
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

    def load(self):
        pass
