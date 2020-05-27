import cv2
import torch
import numpy as np
import torchvision
from typing import Sequence, List

from app.nn_inference.common.base_wrapper import BaseWrapper
from app.base_types import Image
from app.result_types import SegmentationResult


class TorchvisionSegmentationWrapper(BaseWrapper):
    def __init__(self) -> None:
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
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
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
        return torch.cat([preprocessing(image) for image in images], dim=0)

    def predict(self, images: Sequence[Image]) -> List[SegmentationResult]:
        original_size = images[0].shape[0:2]
        original_size = (original_size[1], original_size[0])
        ready_images = self.preprocess(images)
        self.model.eval()
        with torch.no_grad():
            # if len(ready_images.shape) == 3:
            #     predictions = self.model(ready_images.unsqueeze(0))['out']
            #     mask = torch.argmax(predictions.squeeze(), dim=0).cpu().numpy()
            #     mask = cv2.resize(mask.astype(np.uint8), original_size)
            #     return [SegmentationResult(mask)]
            # else:
                predictions = self.model(ready_images.unsqueeze(0))['out']
                masks = torch.argmax(predictions, dim=1).cpu().numpy()
                masks_resized = list(map(lambda img: cv2.resize(img.astype(np.uint8), original_size), masks))
                return [SegmentationResult(mask)
                        for mask in masks_resized]
