import cv2
import sys
import json
import numpy as np
from flask import g
from typing import Dict, Any

sys.path.append('../robot_work_zone_estimation/.')

from robot_work_zone_estimation.src.obj_loader import OBJ
from robot_work_zone_estimation.src.feat_extractor import MakeDescriptor
from robot_work_zone_estimation.src.homography import ComputeHomography
from robot_work_zone_estimation.src.utills import (projection_matrix,
                                                   render, draw_corner)
from app.base_types import Image
from app.nn_inference.faces.wrappers.face_recognition_lib_wrapper import FaceRecognitionLibWrapper
from app.nn_inference.common.utils import draw_bboxes


def read_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as json_file:
        file = json.load(json_file)
    return file


#TODO: implement classes
class DrawRecognition:
    pass


class DrawObjectDetection:
    pass


class DrawSegmentation:
    pass


class DrawFaceDetection:
    def __init__(self, detector: FaceRecognitionLibWrapper) -> None:
        self.detector = detector

    def __call__(self, scene: Image) -> Image:
        det_result = self.detector.get_locations(scene)
        scene = draw_bboxes(scene, det_result)
        return scene


class DrawZone:
    """
    """
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config = read_json(config_path)

        self.obj_file: OBJ = OBJ(self.config['path_to_obj'], swapyz=True)
        self.marker_path = self.config['path_to_marker']
        
        focal_lenght = self.config['focal_lenght']
        frame_width = self.config['frame_width']
        frame_height = self.config['frame_height']
        max_num_of_features = self.config['max_num_of_features']
        scale_factor_feat_det = self.config['scale_factor_feat_det']
        num_of_levels = self.config['num_of_levels']
        self.scale_factor_model = self.config['scale_factor_model']

        self.descriptor_params = dict(nfeatures=max_num_of_features,
                                      scaleFactor=scale_factor_feat_det, 
                                      nlevels=num_of_levels, 
                                      edgeThreshold=10, firstLevel=0)
        self.camera_params = np.array([[focal_lenght , 0,            frame_width//2], 
                                       [0,             focal_lenght, frame_height//2], 
                                       [0,             0,            1]])
        self.column_descriptor = MakeDescriptor(cv2.ORB_create(**self.descriptor_params), 
                                                self.marker_path, 200, 200)
        self.homography_alg = ComputeHomography(cv2.BFMatcher_create(cv2.NORM_HAMMING, 
                                                crossCheck=True))

    def __call__(self, scene: Image) -> Image:
        kp_marker, des_marker = self.column_descriptor.get_marker_data()
        kp_scene, des_scene = self.column_descriptor.get_frame_data(scene)
        if des_marker is not None and des_scene is not None:
            homography = self.homography_alg(kp_scene, kp_marker,
                                             des_scene, des_marker)
            if homography is not None:
                scene = draw_corner(scene,
                                    self.column_descriptor.get_marker_size(),
                                    homography)
                projection = projection_matrix(self.camera_params, homography)
                scene = render(scene, self.obj_file,
                               self.scale_factor_model,
                               projection,
                               self.column_descriptor.get_marker_size(),
                               False)
        return scene


def get_workzone_drawer(config_path: str) -> DrawZone:
    drawer = getattr(g, "_zone_drawer", None)
    if drawer is None:
        drawer = g._zone_drawer = DrawZone(config_path)
    return drawer


def get_face_detection_drawer() -> DrawFaceDetection:
    config = {"model_type": "hog", "number_of_times_to_upsample": 1}
    drawer = getattr(g, "_face_det_drawer", None)
    if drawer is None:
        drawer = g._face_det_drawer = DrawFaceDetection(FaceRecognitionLibWrapper(config))
    return drawer
