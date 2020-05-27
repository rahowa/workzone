import sys
from typing import Tuple

sys.path.append("./robot_work_zone_estimation")

from app.extensions import cache
from app.nn_inference.common.base_wrapper import BaseWrapper
from app.nn_inference.detection.wrappers.detection_wrapper import YOLOWrapper
from app.nn_inference.keypoints.wrappers.torchvision_keypoints_wrapper import TorchvisionKeypointsWrapper
from app.nn_inference.segmentation.wrappers.torchvision_segmentation_wrapper import TorchvisionSegmentationWrapper


@cache.cached()
def get_wrapper(target: str) -> Tuple[BaseWrapper, str]:
    if target == "segmentation":
        wrapper = cache.get("segmentation_wrapper")
        if wrapper is None:
            wrapper = TorchvisionSegmentationWrapper()
            wrapper.load()
            cache.set("segmentation_wrapper", wrapper)
        target_name = "mask"
    elif target == "detection":
        wrapper = cache.get("detection_wrapper")
        if wrapper is None:
            wrapper = YOLOWrapper()
            wrapper.load()
            cache.set("detection_wrapper", wrapper)
        target_name = "boxes"
    else:
        wrapper = cache.get("keypoints_wrapper")
        if wrapper is None:
            wrapper = TorchvisionKeypointsWrapper()
            wrapper.load()
            cache.set("keypoints_wrapper", wrapper)
        target_name = "keypoints"
    return wrapper, target_name
