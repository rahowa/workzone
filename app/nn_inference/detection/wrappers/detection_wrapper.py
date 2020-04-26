import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Sequence, Iterator, List

from app.nn_inference.common.base_wrapper import BaseWrapper
from app.base_types import Image as BaseImage
from app.result_types import DetectionResult

from app.nn_inference.detection.yolo_v3.models import *
from app.nn_inference.detection.yolo_v3.utils.datasets import *
from app.nn_inference.detection.yolo_v3.utils.utils import *

from app.nn_inference.common.utils import chunks


class YOLOWrapper(BaseWrapper):
    def __init__(self, batch_size: int = 2,
                 min_score_tresh: float = 0.8,
                 min_suppression_thres: float = 0.3,
                 img_size: int = 416) -> None:
        current_dir = Path(__file__).parent
        base_path = current_dir.parent / "yolo_v3"
        self.weights_path = base_path / "weights" / "yolov3.weights"
        self.model_config = base_path / "config" / "yolov3.cfg"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tensor_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.min_score_thresh = min_score_tresh
        self.min_suppression_threshold = min_suppression_thres
        self.img_size = img_size
        self.initial_img_size = None

        self.batch_size = batch_size

        self.model = Darknet(self.model_config, self.img_size)

    def __repr__(self):
        return f"YOLOv3 model on {self.device}"

    def load(self) -> bool:
        try:
            self.model.to(self.device)
            #self.model.load_anchors(str(self.anchors_path))
            self.model.load_darknet_weights(str(self.weights_path))
            return True
        except Exception as e:
            print("Loading weights failed", e)
            return False

    def unload(self) -> None:
        self.model.to("cpu")

    def preprocess(self, images: Sequence[BaseImage]) -> Iterator[BaseImage]:
        #print("SHAPE:", images.shape)
        input_imgs = transforms.ToTensor()(images)
        input_imgs, _ = pad_to_square(input_imgs, 0)

        # Resize
        input_imgs = resize(input_imgs, self.img_size)
        input_imgs = input_imgs.type(self.tensor_type)  # Variable(Tensor(input_imgs))
        input_imgs = input_imgs.view(1, input_imgs.shape[0], input_imgs.shape[1], input_imgs.shape[2])

        return (input_imgs)

    def predict(self, images: Sequence[BaseImage]) -> List[DetectionResult]:

        if self.initial_img_size is None:
            self.initial_img_size = images.shape[:2]
            print(self.initial_img_size)
        preprocessed_images = self.preprocess(images)

        with torch.no_grad():
            detections = self.model(preprocessed_images).cpu()
            detections = non_max_suppression(detections, self.min_score_thresh, self.min_suppression_threshold)
            if len(detections) == 0:
                return [DetectionResult()]


        detections = rescale_boxes(detections[0], self.img_size, self.initial_img_size)

        detection_results = []
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            if int(cls_pred) != 0:
                continue

            det_res = DetectionResult()
            det_res.conf = conf
            det_res.boxes = (x1, y1, x2, y2)
            det_res.class_id = cls_pred

            detection_results.append(det_res)

        if len(detection_results) > 0:
            return detection_results
        else:
            return [DetectionResult()]
