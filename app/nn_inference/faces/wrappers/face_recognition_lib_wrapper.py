import numpy as np
import face_recognition as fr
from typing import List, Union, Any, Tuple
from app.nn_inference.common.base_wrapper import BaseWrapper

from app.base_types import Image, Descriptor, Descriptors, Boxes, Box
from app.result_types import FaceResult

class FaceRecognitionLibWrapper(BaseWrapper):
    """
    Wrapper for face_recognition library at
    https://github.com/ageitgey/face_recognition 

    Parameters
    ----------
        config: Union[str, Any]
            Config for face_detection algorithms

    Attributes
    ----------
        config: Dict
            Dictionary with model configuration
        model: str
            Face detection model type ('HOG' or 'CNN')
        self.n_upsample: int
            Upsample factor for hog model. Should be 0 for cnn
    """

    def __init__(self, config: Union[str, Any]):
        if isinstance(config, (str)):
            self.config = self.load_config(config)
        elif isinstance(config, (dict)):
            self.config = config
        else:
            raise NotImplementedError

        self.model = self.config['model_type']
        self.n_upsample = self.config['number_of_times_to_upsample']

    def preprocess(self, image: Image) -> Image:
        """
        Preprocess input image for model
        Parameters
        ----------
            image: Image, np.ndarray
                Image in (WxHxC) format

        Returns
        -------
            image: Image, np.ndarray
                Preprocessed image
        """

        if image.mean() > 1.0:
            return image
        else:
            return np.round(image * 255).astype(np.uint8)

    def convert_boxes(self, box: Tuple[int, int, int, int]) -> Box:
        """Convert box from (ymin xmin ymax xmax) to (xmin ymin xmax ymax)

        Parameters
        ----------
            box: Tuple[int, int, int, int]
                Original wrong box
        Return
        ------
            Bounding box in right format
        """

        return box[1], box[0], box[3], box[2]

    def get_locations(self, data: Image) -> Boxes:
        """
        Returns location of each face at the image

        Parameters
        ----------
            data: Image, np.ndarray
                Image in (WxHxC) format

        Returns
        -------
            face_locations: Bboxes
                List of coordinates of each face
        """

        face_locations = fr.face_locations(data,
                                           number_of_times_to_upsample=self.n_upsample,
                                           model=self.model)

        return tuple(self.convert_boxes(box) for box in face_locations)

    def get_encodings(self, data: Image, bboxes: Boxes) -> Descriptors:
        """
        Returns descriptors for each face at the image

        Parameters
        ----------
            data: Image, np.ndarray
                Image in (WxHxC) format
            bboxes: BBoxes
                Bounding boxes for each face

        Returns
        -------
            face_encodings: Descriptors
                List descriptors for each face
        """

        face_encodings = fr.face_encodings(data, bboxes)
        return face_encodings

    def predict(self, data: Image) -> FaceResult:
        """
        Returns descriptors for each face at the image

        Parameters
        ----------
            data: Image, np.ndarray
                Image in (WxHxC) format

        Returns
        -------
            face_encodings: Descriptors
                List descriptors for each face
        """
        
        data = self.preprocess(data)
        face_locations = self.get_locations(data)
        face_encodings = self.get_encodings(data, face_locations)
        return FaceResult(list(), face_locations, list(), list(face_encodings))

    def match(self, descriptors: Descriptors, sample: Descriptor) -> List[bool]:
        """
        Compute similarity between descriptors and provided sample (also descriptor)

        Parameters
        ----------
            sample: Descriptor, np.ndarray
                Descriptor of current face
            
            descriptors: Descriptors
                All available descriptors to match
            
        Returns
        -------
            matches:
                List of bools where True when sample mathch with descriptor
        """
        return fr.compare_faces(descriptors, sample, tolerance=0.55) 


    def load(self) -> bool:
        return True
    
    def unload(self) -> None:
        pass