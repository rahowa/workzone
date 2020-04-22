import os
import numpy as np
from nptyping import Array
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List

from app.base_types import Boxes
from app.image_decoding_utils import encode_image



class BaseResult(ABC):
    @abstractmethod
    def to_dict(self, path: str) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class FaceResult(BaseResult):
    """
    Result from face detection algotithm for one image

    Parameters
    ----------
        conf: List[float]
            List of confidence scores corresponding to boxes

        boxes: Boxes, Tuple[float, float, float, float]
            List of detected boxes in format of normalized xmin, ymin, xmax, ymax

        landmarks: List[Array[float, 6]], Sequene[np.ndarray]
            List of detected face landmarks corresponding to boxes

    """
    conf: List[float] = field(default_factory=list)
    boxes: Boxes = field(default_factory=tuple)
    landmarks: List[Array[float]] = field(default_factory=list)
    descriptors: List[Array[float]] = field(default_factory=list)

    def to_dict(self, path: str) -> Dict[str, Any]:
        """
        Convert data to dict

        Parameters
        ----------
            path: str
                Path to analyzed image

        Return
        ------
            result dict: Dict[str, Any]
                Dictionary prepared to serizlization
        """

        return {
            "filename": os.path.basename(path),
            "path": path,
            "conf": self.conf,
            "boxes": self.boxes,
            "landmarks": [landmarks for landmarks in self.landmarks]
        }


@dataclass
class DetectionResult(BaseResult):
    """
    Result from object detection algotithm for one image

    Parameters
    ----------
        conf: List[float]
            List of confidence scores corresponding to boxes

        boxes: Boxes, Tuple[float, float, float, float]
            List of detected boxes in format of normalized xmin, ymin, xmax, ymax

        class_id: List[int]
            List of id of detected classes

    """
    conf: List[float] = field(default_factory=list)
    boxes: Boxes = field(default_factory=tuple)
    class_id: List[int] = field(default_factory=list)

    def to_dict(self, path: str) -> Dict[str, Any]:
        """
        Convert data to dict

        Parameters
        ----------
            path: str
                Path to analyzed image

        Return
        ------
            result dict: Dict[str, Any]
                Dictionary prepared to serizlization
        """
        return {
            "filename": os.path.basename(path),
            "path": path,
            "conf": self.conf,
            "boxes": self.boxes,
            "class_id": self.class_id
        }



@dataclass
class SegmentationResult(BaseResult):
    """
    Result from semantic segmentation algotithm for one image

    Parameters
    ----------
        mask: Arra[float]
    """

    mask: Array[float]

    def to_dict(self, path: str) -> Dict[str, Any]:
        """
        Convert data to dict

        Parameters
        ----------
            path: str
                Path to analyzed image

        Return
        ------
            result dict: Dict[str, str]
                Dictionary prepared to serizlization
        """
        return {
            "mask": encode_image((self.mask * 255).astype(np.uint8))
        }