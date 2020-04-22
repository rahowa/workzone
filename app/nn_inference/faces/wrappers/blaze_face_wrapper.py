import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Sequence, Iterator, List

from app.nn_inference.common.base_wrapper import BaseWrapper
from app.base_types import Image, FaceResult
from app.nn_inference.faces.BlazeFace_PyTorch.blazeface import BlazeFace
from app.nn_inference.common.utils import chunks


class BlazeFaceWrapper(BaseWrapper):
    """
    BlazeFace model
    Original implementation at https://github.com/hollance/BlazeFace-PyTorch
    """

    def __init__(self, batch_size: int = 2) -> None:
        current_dir = Path(__file__).parent
        base_path = current_dir.parent / "BlazeFace_PyTorch"
        self.anchors_path = base_path / "anchors.npy"
        self.weights_path = base_path / "blazeface.pth"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BlazeFace()

        self.model.min_score_thresh = 0.75
        self.model.min_suppression_threshold = 0.3

        self.batch_size = batch_size

    def __repr__(self):
        return f"BlazeFace model on {self.device}"

    def load(self) -> bool:
        try:
            self.model.to(self.device)
            self.model.load_anchors(str(self.anchors_path))
            self.model.load_weights(str(self.weights_path))
            return True
        except Exception as e:
            print("Loading weight and anchors failed", e)
            return False

    def preprocess(self, images: Sequence[Image]) -> Iterator[Image]:
        return (cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images)

    def predict(self, images: Sequence[Image]) -> List[FaceResult]:
        ready_images = np.asarray(tuple(self.preprocess(images)))
        if len(ready_images) == 1:
            predictions = self.model.predict_on_image(ready_images[0]).cpu().numpy()
            if predictions.shape[0] > 0:
                return [FaceResult(predictions[16],
                                   tuple(predictions[:, 4], ),
                                   predictions[4:16].tolist()), ]
            else:
                return [FaceResult()]
        else:
            predictions = list()
            batches = chunks(ready_images, self.batch_size)
            for batch in batches:
                predictions.extend(self.model.predict_on_batch(batch))
            return list(map(lambda pred: FaceResult(pred[:, 16].tolist(),
                                                    pred[:, 0:4].tolist(),
                                                    pred[:, 4:16].tolist())
                                         if pred.shape[0] > 0
                                         else FaceResult(), predictions))
