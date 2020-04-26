import torch
import torchvision
from typing import Sequence, List

from app.nn_inference.common.base_wrapper import BaseWrapper
from app.base_types import Image
from app.result_types import KeypointsResult


class TorchvisionKeypointsWrapper(BaseWrapper):
    def __init__(self) -> None:
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),    
            torchvision.transforms.Resize((320, 240)),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
        return preprocessing(images)

    def predict(self, images: Sequence[Image]) -> List[KeypointsResult]:
        ready_images = self.preprocess(images)
        self.model.eval()
        with torch.no_grad():
            if len(ready_images.shape) == 3:
                predictions = self.model(ready_images.unsqueeze(0))[0]
                boxes = predictions["boxes"]
                keypoints = predictions["keypoints"]
                return [KeypointsResult(boxes, keypoints)]
            else:   
                predictions = self.model(ready_images.unsqueeze(0))
                boxes = [image_pred["boxes"] for image_pred in predictions]
                keypoints = [image_pred["keypoints"] for image_pred in predictions]
                return [KeypointsResult(img_boxes, img_keypoints)
                        for img_boxes, img_keypoints in zip(boxes, keypoints)]
