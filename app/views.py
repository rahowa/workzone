import jsonpickle
from flask import request
from typing import Any, Union
from flask import Response, Blueprint

from app.extensions import mongo
from app.image_decoding_utils import deocode_image
from app.fill_databse import FillDatabase
from app.mongo_controller import MongoController
from app.route_utils import load_classes
from app.faces_utils import check_persons, detect_faces
from app.nn_inference.faces.wrappers.face_recognition_lib_wrapper import FaceRecognitionLibWrapper
from app.nn_inference.segmentation.wrappers.torchvision_segmentation_wrapper import TorchvisionSegmentationWrapper


bp_main = Blueprint('blueprint_main', __name__)


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
        response = {'faces_id': str(recognition_result)}
        response = jsonpickle.encode(response)
        return Response(response, status=200, mimetype="application/json")
    elif target == "detect":
        face_locations = detect_faces(image)
        response = {'faces_loc': str(face_locations)}
        response = jsonpickle.encode(response)
        return Response(response, status=200, mimetype="application/json")
    else:
        return Response("empty", status=404)


@bp_main.route("/object/<target>")
def objects_processing(target: str) -> Response:
    """
    Apply object detection or segmentation on on image
    Parameters
    ----------
        target: str
            'segmentation' for semantic segmentation. Return response with RLE masks.
            'detection' for object detection. Return response with detected boxes and classes.
    Return
    ------
        Response with target result
    """
    image = deocode_image(request.data)
    if target == "segmentation":
        wrapper = TorchvisionSegmentationWrapper()
        res = wrapper.predict((image,))[0]
        response = {'mask': res.to_dict(f"/object/{target}")}
        response = jsonpickle.encode(response)
        return Response(response, status=200, mimetype="application/json")
    else:
        return Response("Not available", status=404, mimetype="application/json")



@bp_main.route("/workzone")
def compute_workzone():
    pass


@bp_main.route("/fill_db")
def fill_database() -> str:
    """
    Fill database with descriptors extracted from images
    """
    controller = MongoController(mongo)
    face_det = FaceRecognitionLibWrapper("./nn_inference/configs/test_fr_hog_config.json")
    FillDatabase(controller, face_det)("./face_database")
    return "Workers updated"


@bp_main.route('/classes/<target>')
def get_availabel_classes(target: str) -> Response:
    """
    Return Json with all available object detection and segmentation classes
    """

    if target == "all":
        return load_classes("classes.txt")
    elif target == "segmentation":
        return load_classes("segmentation_classes.txt")
    else:
        return Response("WRONG TARGET")