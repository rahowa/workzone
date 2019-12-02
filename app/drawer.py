import cv2
import sys 
import json
import numpy as np 
from typing import Dict, Any

sys.path.append('~/Training/workzone/robot_work_zone_estimation/.')

from robot_work_zone_estimation.src.obj_loader import OBJ 
from robot_work_zone_estimation.src.feat_extractor import MakeDescriptor
from robot_work_zone_estimation.src.homography import ComputeHomography
from robot_work_zone_estimation.src.utills import (projection_matrix, 
                        render, draw_corner)
from .camera import Image

def read_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as json_file:
        file = json.load(json_file)
    return file 


class DrawZone:
    """
    """
    def __init__(self, config_path: str):
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
    def __call__(self, scene: Image):
        # _, scene  = cap.read()
        kp_marker, des_marker = self.column_descriptor.get_marker_data()
        kp_scene, des_scene = self.column_descriptor.get_frame_data(scene)
        if des_marker is not None and des_scene is not None:
            homography = self.homography_alg(kp_scene, kp_marker, des_scene, des_marker)

            if homography is not None:
                scene = draw_corner(scene, self.column_descriptor.get_marker_size(), homography)
                projection = projection_matrix(self.camera_params, homography)
                scene = render(scene, self.obj_file, self.scale_factor_model, 
                                projection, self.column_descriptor.get_marker_size(), False)
        
            # if self.homography_alg.matches is not None:
            #     scene = cv2.drawMatches(self.column_descriptor.marker,
            #                             kp_marker,
            #                             scene, 
            #                             kp_scene, 
            #                             self.homography_alg.matches, 
            #                             0, flags=2)
        return scene