from pathlib import Path
from typing import Sequence, Iterator, List

import mxnet as mx
from gluoncv import model_zoo, data

from app.nn_inference.common.base_wrapper import BaseWrapper
from app.base_types import Box, Image as BaseImage
from app.result_types import DetectionResult


class HelmetnetWrapper(BaseWrapper):
    def __init__(self, batch_size: int = 2,
                 min_score_tresh: float = 0.8,
                 min_suppression_thres: float = 0.3,
                 img_size: int = 416) -> None:
        current_dir = Path(__file__).parent

        self.helmetnet_weights_path = current_dir.parent / "helmetnet" / "weights" / "darknet.params"
        self.ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
        self.device = "cuda" if mx.context.num_gpus() > 0 else "cpu"

        self.min_score_thresh = min_score_tresh
        self.min_suppression_threshold = min_suppression_thres
        self.img_size = img_size
        self.initial_img_size = None
        self.batch_size = batch_size
        helmet_network_type = "yolo3_darknet53_voc"  # "yolo3_mobilenet1.0_voc" "mobilenet0.25.params"
        self.helmetnet_model = model_zoo.get_model(helmet_network_type,
                                                   pretrained=False)

    def __repr__(self):
        return f"HELMETNET model on {self.device}"

    def load(self) -> bool:
        try:
            """Initialize helmet detection network"""
            classes = ['hat', 'person']
            for param in self.helmetnet_model.collect_params().values():
                if param._data is not None:
                    continue
                param.initialize()
            self.helmetnet_model.reset_class(classes)
            self.helmetnet_model.collect_params().reset_ctx(self.ctx)
            self.helmetnet_model.load_parameters(str(self.helmetnet_weights_path), ctx=self.ctx)

            return True

        except Exception as e:
            print("Loading weights failed", e)
            return False

    def unload(self) -> None:
        # self.model.to("cpu")
        pass

    def preprocess(self, image: Sequence[BaseImage]) -> Iterator[BaseImage]:
        # mx_images = list(map(lambda img: mx.nd.array(img), image))
        return data.transforms.presets.yolo.transform_test([mx.nd.array(image)])[0]

    def xyhw_to_xyxy(self, xywh) -> Box:
        w, h = max(xywh[2] - 1, 0), max(xywh[3] - 1, 0)
        return xywh[0], xywh[1], xywh[0] + w, xywh[1] + h

    def predict_on_image(self, preprocessed_image: BaseImage) -> DetectionResult:
        x = preprocessed_image.as_in_context(self.ctx)
        helmetnet_class_ids, helmetnet_scores, helmetnet_bboxes = self.helmetnet_model(x)

        helmetnet_class_ids = helmetnet_class_ids.asnumpy()[0].reshape(1, -1).astype("int").tolist()[0]
        helmetnet_scores = helmetnet_scores.asnumpy()[0].reshape(1, -1).tolist()[0]
        helmetnet_bboxes = helmetnet_bboxes.asnumpy()[0].reshape(-1, 4).astype("int").tolist()

        helmetnet_class_ids = list(filter(lambda idx: idx != -1.0, helmetnet_class_ids))
        helmetnet_scores = list(filter(lambda conf: conf != -1.0, helmetnet_scores))
        helmetnet_bboxes = list(filter(lambda box: box[0] != -1.0, helmetnet_bboxes))
        helmetnet_bboxes = list(map(lambda box: tuple(self.xyhw_to_xyxy(box)), helmetnet_bboxes))
        return DetectionResult(helmetnet_scores,
                               helmetnet_bboxes,
                               helmetnet_class_ids)

    def predict(self, images: Sequence[BaseImage]) -> List[DetectionResult]:

        '''In case of several different streams images sizes can be different
        but now it is assumed that they are equivalent'''
        if self.initial_img_size is None:
            self.initial_img_size = images[0].shape[:2]

        detection_results = []

        if len(images) == 1:
            images = images[0]

            preprocessed_image = self.preprocess(images)
            print(preprocessed_image)  # everything fails without it
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
