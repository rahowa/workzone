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
        det_result = self.detector.predict(scene)
        bboxes = [det.boxes for det in det_result if det.boxes != ()]
        scene = draw_bboxes(scene, bboxes)
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


def get_workzone_drawer() -> DrawArucoZone:
    config_path = "../robot_work_zone_estimation/aruco_config.json"
    drawer = getattr(g, "_zone_drawer", None)
    if drawer is None:
        drawer = g._zone_drawer = DrawArucoZone(config_path)
    return drawer


def get_face_detection_drawer() -> DrawFaceDetection:
    config = {"model_type": "cnn", "number_of_times_to_upsample": 0}
    drawer = getattr(g, "_face_det_drawer", None)
    if drawer is None:
        drawer = g._face_det_drawer = DrawFaceDetection(FaceRecognitionLibWrapper(config))
    return drawer


def get_segmentation_drawer() -> DrawSegmentation:
    drawer = getattr(g, "_segmentation_drawer", None)
    if drawer is None:
        drawer = g._face_det_drawer = DrawSegmentation(TorchvisionSegmentationWrapper())
    return drawer


def get_object_detection_drawer() -> DrawObjectDetection:
    drawer = getattr(g, "_obj_det_drawer", None)
    if drawer is None:
        drawer = g._obj_det_drawer = DrawObjectDetection(YOLOWrapper())
    return drawer


def get_keypoints_drawer() -> DrawKeypoints:
    drawer = getattr(g, "_keypoints_drawer", None)
    if drawer is None:
        drawer = g._face_det_drawer = DrawKeypoints(TorchvisionKeypointsWrapper())
    return drawer
