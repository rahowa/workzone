import sys
import json
import numpy as np
from typing import Any, Union
from flask import Response, Blueprint, request

sys.path.append("./robot_work_zone_estimation")

from app.extensions import mongo, cache
from app.route_utils import load_classes
from app.fill_databse import FillDatabase
from app.objects_utils import get_wrapper
from app.mongo_controller import MongoController
from app.image_decoding_utils import deocode_image
from app.faces_utils import check_persons, detect_faces
from robot_work_zone_estimation.src.workzone import Workzone
from robot_work_zone_estimation.src.calibrate_camera_utils import CameraParams
from app.nn_inference.faces.wrappers.face_recognition_lib_wrapper import FaceRecognitionLibWrapper
from robot_work_zone_estimation.src.aruco_zone_estimation import ArucoZoneEstimator, ARUCO_MARKER_SIZE


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
        db_controller = MongoController(mongo)
        face_locations = detect_faces(image)
        recognition_result = check_persons(image, db_controller)
        response = {'faces_id': list(recognition_result),
                    'faces_loc': list(face_locations)
                    }
        response = json.dumps(response)
        return Response(response, status=200, mimetype="application/json")
    elif target == "detect":
        face_locations = detect_faces(image)
        response = {'faces_loc': list(face_locations)}
        response = json.dumps(response)
        return Response(response, status=200, mimetype="application/json")
    else:
        return Response("empty", status=404)


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
    wrapper, target_name = get_wrapper(target)
    res = wrapper.predict((image, ))[0]
    response = {target_name: res.to_dict(f"/object/{target}")}
    response = json.dumps(response)
    return Response(response, status=200, mimetype="application/json")


@cache.cached(timeout=350, key_prefix="zone_estimator")
def get_zone_estimator():

    aruco_params = cache.get("aruco_params")
    if aruco_params is None:
        with open("robot_work_zone_estimation/aruco_config.json") as conf_file:
            aruco_params = json.load(conf_file)
        cache.set("aruco_params", aruco_params)

    marker_id = aruco_params["marker_idx"]
    wz_cx = aruco_params["wz_cx"]
    wz_cy = aruco_params["wz_cy"]
    wz_height = aruco_params["wz_height"]
    wz_width = aruco_params["wz_width"]
    marker_world_size = aruco_params["marker_world_size"]
    marker_size = aruco_params["marker_size"]
    camera_params = aruco_params["camera_params"]
    camera_params = CameraParams(np.array(camera_params['camera_mtx'],
                                          dtype=np.float),
                                 np.array(camera_params['distortion_vec'],
                                          dtype=np.float),
                                 np.array(camera_params['rotation_vec'],
                                          dtype=np.float),
                                 np.array(camera_params['translation_vec'],
                                          dtype=np.float))

    zone = Workzone(wz_cx, wz_cy, wz_height, wz_width)
    estimator = cache.get("zone_estimator")
    if estimator is None:
        estimator = ArucoZoneEstimator(marker_world_size,
                                       ARUCO_MARKER_SIZE[marker_size],
                                       marker_id,
                                       camera_params,
                                       zone)
        cache.set("zone_estimator", estimator)
    return estimator


@bp_main.route("/workzone", methods=['POST'])
def compute_workzone():
    image = deocode_image(request.data)
    estimator = get_zone_estimator()
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
