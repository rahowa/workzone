from app.nn_inference.common.base_wrapper import BaseWrapper
from os import rename
import cv2
import sys
import json
import numpy as np
from flask import g
from typing import Dict, Any

sys.path.append('../robot_work_zone_estimation/.')

from robot_work_zone_estimation.src.workzone import Workzone
from robot_work_zone_estimation.src.calibrate_camera_utils import CameraParams
from robot_work_zone_estimation.src.aruco_zone_estimation import ArucoZoneEstimator, ARUCO_MARKER_SIZE

from app.base_types import Image
from app.extensions import cache
from app.nn_inference.detection.wrappers.detection_wrapper import YOLOWrapper
from app.nn_inference.common.utils import draw_bboxes, decode_segmap, draw_keypoints
from app.nn_inference.faces.wrappers.face_recognition_lib_wrapper import FaceRecognitionLibWrapper
from app.nn_inference.keypoints.wrappers.torchvision_keypoints_wrapper import TorchvisionKeypointsWrapper
from app.nn_inference.segmentation.wrappers.torchvision_segmentation_wrapper import TorchvisionSegmentationWrapper


def read_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as json_file:
        file = json.load(json_file)
    return file


class DrawKeypoints:
    def __init__(self, detector: TorchvisionKeypointsWrapper) -> None:
        self.detector = detector

    def __call__(self, scene: Image) -> Image:
        det_result = self.detector.predict(scene)[0]
        boxes = det_result.boxes
        keypoints = det_result.keypoints
        scene = draw_keypoints(scene, keypoints)
        scene = draw_bboxes(scene, boxes)
        return scene


class DrawObjectDetection:
    def __init__(self, detector: YOLOWrapper) -> None:
        self.detector = detector
        self.detector.load()

    def __call__(self, scene: Image) -> Image:
        det_results = self.detector.predict((scene, ))

        if not (len(det_results) == 1 and len(det_results[0].boxes) == 0):
            #for det_result in det_results:
            #    scene = draw_bboxes(scene, det_result.boxes)
            scene = draw_bboxes(scene, det_results[0].boxes)

        return scene


class DrawSegmentation:
    def __init__(self, detector: TorchvisionSegmentationWrapper) -> None:
        self.detector = detector

    def __call__(self, scene: Image) -> Image:
        det_result = self.detector.predict((scene, ))[0].mask
        scene = decode_segmap(det_result)
        return scene


class DrawFaceDetection:
    def __init__(self, detector: FaceRecognitionLibWrapper) -> None:
        self.detector = detector

    def __call__(self, scene: Image) -> Image:
        det_result = self.detector.get_locations(scene)
        scene = draw_bboxes(scene, det_result)
        return scene


class DrawArucoZone:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config = read_json(config_path)

        marker_id = self.config["marker_idx"]
        marker_world_size = self.config["marker_world_size"]
        marker_size = ARUCO_MARKER_SIZE[self.config["marker_size"]]
        camera_params_dict = self.config["camera_params"]
        camera_params = CameraParams(np.array(camera_params_dict["camera_mtx"]),
                                     np.array(camera_params_dict["distortion_vec"]),
                                     np.array(camera_params_dict["rotation_vec"]),
                                     np.array(camera_params_dict["translation_vec"]))
        wz_cx = self.config["wz_cx"]
        wz_cy = self.config["wz_cy"]
        wz_height = self.config["wz_height"]
        wz_width = self.config["wz_width"]

        zone = Workzone(wz_cx, wz_cy, wz_height, wz_width)
        self.estimator = ArucoZoneEstimator(marker_world_size,
                                            marker_size,
                                            marker_id,
                                            camera_params,
                                            zone)

    def __call__(self, scene: Image) -> Image:
        zone_polygon = self.estimator.estimate(scene)
        zone_polygon = np.array(zone_polygon).reshape(-1, 1, 2)
        zone_polygon = np.clip(zone_polygon, 0, np.inf)
        scene = cv2.polylines(scene, [np.int32(zone_polygon)], True, (255, 0, 0), 2, cv2.LINE_AA)
        return scene


@cache.cached(100)
def get_workzone_drawer() -> DrawArucoZone:
    config_path = "../robot_work_zone_estimation/aruco_config.json"
    drawer = cache.get("workzone_drawer")
    if drawer is None:
        drawer = DrawArucoZone(config_path)
        cache.set("workzone_drawer", drawer)
    return drawer


@cache.cached(100)
def get_face_detection_drawer() -> DrawFaceDetection:
    config = {"model_type": "cnn", "number_of_times_to_upsample": 0}
    drawer = cache.get("face_det_drawer")
    if drawer is None:
        drawer = DrawFaceDetection(FaceRecognitionLibWrapper(config))
        cache.set("face_det_drawer", drawer)
    return drawer


@cache.cached(100)
def get_segmentation_drawer() -> DrawSegmentation:
    drawer = cache.get("segmentation_drawer")
    if drawer is None:
        drawer = DrawSegmentation(TorchvisionSegmentationWrapper())
        cache.set("segmentation_drawer", DrawSegmentation(TorchvisionSegmentationWrapper()))
    return drawer


@cache.cached(100)
def get_object_detection_drawer() -> DrawObjectDetection:
    drawer = cache.get("obj_det_drawer")
    if drawer is None:
        drawer = DrawObjectDetection(YOLOWrapper())
        cache.set("obj_det_drawer", drawer)
    return drawer


@cache.cached(100)
def get_keypoints_drawer() -> DrawKeypoints:
    drawer = cache.get("keypoints_drawer")
    if drawer is None:
        drawer = DrawKeypoints(TorchvisionKeypointsWrapper())
        cache.set("keypoints_drawer", drawer)
    return drawer
