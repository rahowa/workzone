import torch
from pathlib import Path
from typing import Sequence, Iterator, List

from app.nn_inference.common.base_wrapper import BaseWrapper
from app.base_types import Image as BaseImage
from app.result_types import DetectionResult

from app.nn_inference.detection.yolo_v3.models import *
from app.nn_inference.detection.yolo_v3.utils.datasets import *
from app.nn_inference.detection.yolo_v3.utils.utils import *


HELMET_DETECT_EVENT = 2


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
            """Initialize basic object detection network"""
            self.model.to(self.device)
            self.model.load_darknet_weights(str(self.weights_path))
            return True

        except Exception as e:
            print("Loading weights failed", e)
            return False

    def unload(self) -> None:
        self.model.to("cpu")

    def preprocess(self, image: Sequence[BaseImage]) -> Iterator[BaseImage]:
        input_imgs = transforms.ToTensor()(image)
        input_imgs, _ = pad_to_square(input_imgs, 0)

        # Resize
        input_imgs = resize(input_imgs, self.img_size)
        input_imgs = input_imgs.type(self.tensor_type)  # Variable(Tensor(input_imgs))
        input_imgs = input_imgs.view(1, input_imgs.shape[0], input_imgs.shape[1], input_imgs.shape[2])

        return (input_imgs)

    def predict_on_image(self, preprocessed_image: BaseImage) -> List[DetectionResult]:
        '''Detect objects'''
        with torch.no_grad():
            detections = self.model(preprocessed_image).cpu()
            detections = non_max_suppression(detections,
                                             self.min_score_thresh,
                                             self.min_suppression_threshold)
            if len(detections) == 0:
                return [DetectionResult()]
        detections = rescale_boxes(detections[0], self.img_size, self.initial_img_size)

        boxes = list()
        confs = list()
        class_ids = list()
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            # Validate only persons
            if int(cls_pred) != 0:
                continue

            confs.append(float(conf.numpy()))
            boxes.append((int(x1.numpy()), int(y1.numpy()),
                          int(x2.numpy()), int(y2.numpy())))
            class_ids.append(int(cls_pred.numpy()))

        det_res = DetectionResult(boxes=tuple(boxes), class_id=class_ids, conf=confs)
        return det_res

    def predict(self, images: Sequence[BaseImage]) -> List[DetectionResult]:

        '''In case of several different streams images sizes can be different
        but now it is assumed that they are equivalent'''
        if self.initial_img_size is None:
            self.initial_img_size = images[0].shape[:2]

        detection_results = []

        if len(images) == 1:
            images = images[0]

            preprocessed_image = self.preprocess(images)
            det_res = self.predict_on_image(preprocessed_image)
            if len(det_res.boxes) > 0:
                detection_results.append(det_res)
            else:
                detection_results.append(DetectionResult())
        else:
            for image in images:
                preprocessed_image = self.preprocess(image)
                det_res = self.predict_on_image(preprocessed_image)

                if len(det_res.boxes) > 0:
                    detection_results.append(det_res)
                else:
                    detection_results.append(DetectionResult())
        return detection_results
