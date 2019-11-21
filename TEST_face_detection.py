from collections import deque
from abc import ABC, abstractmethod
from typing import Tuple, NewType, Dict, Union, TypeVar, Any

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]
BBoxes = Tuple[BBox, ...]
T = TypeVar('T')
Optional = NewType("Optional", Union[T, None])
Eyes = NewType("Eyes", Union[Dict[str, Union[BBox, None]], None])


class Detector(ABC):
    @abstractmethod
    def predict(self, *args, **kwargs) -> Tuple[bool, BBox]:
        raise NotImplementedError

    @abstractmethod
    def enchance_image(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class CascadeFaceDetector(Detector):
    def __init__(self, weights_path: str):
        # super().__init__()
        self.weights_path = weights_path
        self.detector = cv2.CascadeClassifier(weights_path)

    def predict(self, image: np.ndarray) -> Tuple[bool, BBox]:
        image = self.enchance_image(image)
        bboxes = self.detector.detectMultiScale(image)
        if len(bboxes) > 0:
            return True, bboxes[0]
        else:
            return False, (0, 0, image.shape[0], image.shape[1])

    def enchance_image(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
