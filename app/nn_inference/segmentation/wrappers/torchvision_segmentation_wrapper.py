import torch
import numpy as np
import torchvision
from typing import Optional, Sequence, List

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
        return preprocessing(images)

    def predict(self, images: Sequence[Image]) -> List[SegmentationResult]:
        ready_images = self.preprocess(images)
        self.model.eval()
        with torch.no_grad():
            if len(ready_images.shape) == 3:
                predictions = self.model(ready_images.unsqueeze(0))['out']
                mask = torch.argmax(predictions.squeeze(), dim=0).cpu().numpy()
                print("MASK SHAPE ", mask.shape)
                return [SegmentationResult(mask)]
            else:   
                predictions = self.model(ready_images.unsqueeze(0))['out']
                mask = torch.argmax(predictions, dim=1).cpu().numpy()
                print("MASK SHAPE ", mask.shape)
                return [SegmentationResult(mask[idx])
                        for idx in mask.shape[0]]
