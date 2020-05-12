import sys
import json
import numpy as np
from typing import Any, Union
from flask import Response, Blueprint, request

sys.path.append("./robot_work_zone_estimation")

from app.extensions import mongo
from app.route_utils import load_classes
from app.fill_databse import FillDatabase
from app.mongo_controller import MongoController
from app.image_decoding_utils import deocode_image
from app.faces_utils import check_persons, detect_faces
from app.nn_inference.detection.wrappers.detection_wrapper import YOLOWrapper
from app.nn_inference.faces.wrappers.face_recognition_lib_wrapper import FaceRecognitionLibWrapper
from app.nn_inference.segmentation.wrappers.torchvision_segmentation_wrapper import TorchvisionSegmentationWrapper
from app.nn_inference.keypoints.wrappers.torchvision_keypoints_wrapper import TorchvisionKeypointsWrapper
from robot_work_zone_estimation.src.aruco_zone_estimation import ArucoZoneEstimator, ARUCO_MARKER_SIZE
from robot_work_zone_estimation.src.workzone import Workzone
from robot_work_zone_estimation.src.calibrate_camera import CameraParams


bp_main = Blueprint('blueprint_main', __name__)


# TODO: cache wrappers
@bp_main.route('/face/<target>', methods=['POST'])
def face_processing(target: str) -> Union[Response, Any]:
    """
    Detect or recognize faces on image
    Parameters
    ----------
        target: str
            'detect' for face detection. Return response with boxes.
            'recognize' for face recognition. Return response with recognized ids.
    Return
    ------
        Response with target result
    """

    image = deocode_image(request.data)
    if target == "recognize":
        db_controller = MongoController(mongo)  # TODO: cache results
        recognition_result = check_persons(image, db_controller)
        response = {'faces_id': list(recognition_result)}
        response = json.dumps(response)
        return Response(response, status=200, mimetype="application/json")
    elif target == "detect":
        face_locations = detect_faces(image)
        response = {'faces_loc': list(face_locations)}
        response = json.dumps(response)
        return Response(response, status=200, mimetype="application/json")
    else:
        return Response("empty", status=404)


# TODO: cache wrappers
@bp_main.route("/object/<target>", methods=['POST'])
def objects_processing(target: str) -> Response:
    """
    Apply object detection or segmentation on on image
    Parameters
    ----------
        target: str
            'segmentation' for semantic segmentation. Return response with RLE masks.
            'detection' for object detection. Return response with detected boxes and classes.
            'keypoints' for human keypoints detection. Return responce with [x, y] coordinates
            of each keypoint.
    Return
    ------
        Response with target result
    """

    image = deocode_image(request.data)
    if target == "segmentation":
        wrapper = TorchvisionSegmentationWrapper()
        wrapper.load()
        res = wrapper.predict((image,))[0]
        response = {'mask': res.to_dict(f"/object/{target}")}
        response = json.dumps(response)
        return Response(response, status=200, mimetype="application/json")
    elif target == "detection":
        wrapper = YOLOWrapper()
        wrapper.load()
        res = wrapper.predict((image,))[0]
        response = {'boxes': res.to_dict(f"/object/{target}")}
        response = json.dumps(response)
        return Response(response, status=200, mimetype="application/json")
    else:
        wrapper = TorchvisionKeypointsWrapper()
        wrapper.load()
        res = wrapper.predict(image)[0]
        response = {"keypoints": res.to_dict(f"/object/{target}")}
        response = json.dumps(response)
        return Response(response, status=200, mimetype="application/json")


# TODO: cache aruco params
@bp_main.route("/workzone", methods=['POST'])
def compute_workzone():
    with open("robot_work_zone_estimation/aruco_config.json") as conf_file:
        aruco_params = json.load(conf_file)

    wz_cx = aruco_params["wz_cx"]
    wz_cy = aruco_params["wz_cy"]
    wz_height = aruco_params["wz_height"]
    wz_width = aruco_params["wz_width"]
    marker_world_size = aruco_params["marker_world_size"]
    marker_size = aruco_params["marker_size"]
    camera_params = aruco_params["camera_params"]
    camera_params = CameraParams(np.array(camera_params['camera_mtx'], dtype=np.float),
                                 np.array(camera_params['distortion_vec'], dtype=np.float),
                                 np.array(camera_params['rotation_vec'], dtype=np.float),
                                 np.array(camera_params['translation_vec'], dtype=np.float))

    image = deocode_image(request.data)
    zone = Workzone(wz_cx, wz_cy, wz_height, wz_width)
    estimator = ArucoZoneEstimator(marker_world_size,
                                   ARUCO_MARKER_SIZE[marker_size],
                                   camera_params,
                                   zone)
    zone_polygon = estimator.estimate(image)
    response = json.dumps({"workzone": zone_polygon})
    return Response(response, status=200, mimetype="application/json")


@bp_main.route("/fill_db")
def fill_database() -> str:
    """
    Fill database with descriptors extracted from images
    """

    controller = MongoController(mongo)
    config = {"model_type": "cnn", "number_of_times_to_upsample": 0}

    face_det = FaceRecognitionLibWrapper(config)
    FillDatabase(controller, face_det)("./face_database")
    return "<p>Workers updated</p>"


# TODO : cachle loading
@bp_main.route('/classes/<target>')
def get_availabel_classes(target: str) -> Response:
    """
    Return Json with all available object detection and segmentation classes
    """

    if target == "all":
        return load_classes("classes.txt")
    elif target == "segmentation":
        return load_classes("segmentation_classes.txt")
    elif target == "detection":
        return load_classes("detection_classes.txt")
    elif target == "keypoints":
        return load_classes("keypoints_classes.txt")
    else:
        return Response("WRONG TARGET")
