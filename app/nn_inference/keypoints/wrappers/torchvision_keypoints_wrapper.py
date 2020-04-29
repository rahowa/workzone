import torch
import numpy as np
import torchvision
from typing import Sequence, List

from app.nn_inference.common.base_wrapper import BaseWrapper
from app.base_types import Image
from app.result_types import KeypointsResult


class TorchvisionKeypointsWrapper(BaseWrapper):
    def __init__(self) -> None:
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.x_inp_ratio = 800/640
        self.y_inp_ration = 800/480
        self.kp_result_normalizatoin = np.array([self.x_inp_ratio, self.y_inp_ration, 1.0]).reshape(1, 1, 3)
        self.box_result_normalization = np.array([self.x_inp_ratio, self.y_inp_ration,
                                                  self.x_inp_ratio, self.y_inp_ration]).reshape(1, 4)

    def __repr__(self):
        return f"DeeplabV3 (ResNet50) model on {self.device}"

    def load(self) -> bool:
        try:
            self.model.to(self.device)
            return True
        except Exception as e:
            print("Loading weight and anchors failed", e)
            return False

    def unload(self) -> None:
        self.model.to("cpu")
    
    def preprocess(self, images: Sequence[Image]) -> torch.Tensor:
        preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),    
            torchvision.transforms.Resize((int(self.x_inp_ratio * 640), int(self.y_inp_ration * 480))),
            torchvision.transforms.ToTensor(),
        ])
        return preprocessing(images)

    def predict(self, images: Sequence[Image]) -> List[KeypointsResult]:
        ready_images = self.preprocess(images)
        self.model.eval()
        with torch.no_grad():
            if len(ready_images.shape) == 3:
                predictions = self.model(ready_images.unsqueeze(0))[0]
                boxes = predictions["boxes"].cpu().numpy() / self.box_result_normalization
                keypoints = predictions["keypoints"].cpu().numpy() / self.kp_result_normalizatoin
                return [KeypointsResult(boxes.tolist(), keypoints)]
            else:   
                predictions = self.model(ready_images.unsqueeze(0)).cpu().numpy()
                boxes = [image_pred["boxes"].cpu().numpy() / self.box_result_normalization
                         for image_pred in predictions]
                keypoints = [image_pred["keypoints"].cpu().numpy() / self.kp_result_normalizatoin
                             for image_pred in predictions]
                return [KeypointsResult(img_boxes.tolist(), img_keypoints)
                        for img_boxes, img_keypoints in zip(boxes, keypoints)]
