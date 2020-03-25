# from collections import deque
from typing import Any, Tuple, Union
from dataclasses import dataclass

import cv2
import jsonpickle
import numpy as np
from nptyping import Array
from flask import Response, Blueprint
from flask import request, render_template

from app.camera import Image
from app.extensions import mongo
from app.fill_databse import FillDatabase
from app.mongo_controller import MongoController

from app.base_types import BBoxes
from nn_inference.external_wrappers import FaceRecognitionLibWrapper


bp_main = Blueprint('blueprint_main', __name__)


@dataclass
class DetectionResult:
    image: Array[int]
    labels: Tuple[Union[int, str], ...]
    bboxes: BBoxes
    

def recognize_face(frame: Image, db_controller: MongoController) -> DetectionResult:
    known_face_encodings = db_controller.all_valid_descriptors()
    if len(known_face_encodings) == 0:
        return DetectionResult(frame, (-1, ), ((0, 0, 0, 0), ))

    face_det = FaceRecognitionLibWrapper("./nn_inference/configs/test_fr_hog_config.json")
    face_locations = face_det.get_locations(frame)
    face_encodings = face_det.get_encodings(frame, face_locations)
    labels = list()
    for face_encoding in face_encodings:
        label = -1
        matches = face_det.match(face_encoding, known_face_encodings)
        if True in matches:
            label = matches.index(True)
        labels.append(label)
    return DetectionResult(frame, tuple(labels), face_locations)


@bp_main.route('/face', methods=['GET', 'POST'])
def face_recognition_pipeline() -> Union[Response, Any]:
    if request.method == 'POST':
        image = np.fromstring(request.data, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        db_controller = MongoController(mongo)
        detection_result = recognize_face(image, db_controller)
        response = {'message': f"{detection_result.labels} was detected"}
        response = jsonpickle.encode(response)
        return Response(response, status=200, mimetype="application/json")
    else:
        return render_template("session_history.html")


@bp_main.route("/fill_db")
def fill_database() -> str:
    controller = MongoController(mongo)
    face_det = FaceRecognitionLibWrapper("./nn_inference/configs/test_fr_hog_config.json")
    FillDatabase(controller, face_det)("./face_database")
    return "Workers updated"
