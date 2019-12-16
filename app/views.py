from typing import List, Union
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import jsonpickle
from flask import request, render_template, Response, Blueprint
import face_recognition as fr

from .camera import Image
from .extensions import mongo
from .mongo_controller import MongoController
from .fill_databse import FillDatabase

from nn_inference.base_wrapper import BBoxes
from nn_inference.external_wrappers import FaceDetectionLibWrapper


main = Blueprint('main', __name__)

@dataclass
class DetectionResult:
    image: np.ndarray
    labels: List[Union[int, str]]
    bboxes: BBoxes


def PLACEHOLDER_detect_objects(image: Image) -> DetectionResult:
    """
    """
    return DetectionResult(image, "human", ((20, 20, 50, 50)))


def PLACEHOLDER_recognaize_face(image: Image,
                                db_controller: MongoController) -> DetectionResult:
    """
    """
    face_locations = fr.face_locations(image)
    face_encodings = fr.face_encodings(image, face_locations)

    print("Getting all descriptors... ".center(79))
    known_face_encodings = db_controller.all_descriptors()
    print("All descriptors are loaded".center(79))
    labels = deque(maxlen=len(face_encodings))

    print(f"num of known face encodings: \
        {len(known_face_encodings)}".center(79))

    if len(known_face_encodings) == 0:
        return DetectionResult(image, [-1], [0, 0, 1, 1])

    for face_encoding in face_encodings:
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        label = -1
        if True in matches:
            label = matches.index(True)
        labels.append(label)
    return DetectionResult(image, labels, face_locations)


@main.route('/face', methods=['GET', 'POST'])
def face_recognition_pipeline():
    if request.method == 'POST':
        image = np.fromstring(request.data, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        print('Recvie image'.center(79))
        db_controller = MongoController(mongo)
        print("Connect to databse".center(79))
        detection_result = PLACEHOLDER_recognaize_face(image, db_controller)
        response = {'message': f"{detection_result.labels} was detected"}
        response = jsonpickle.encode(response)
        return Response(response, status=200, mimetype="application/json")
    else:
        return render_template("session_history.html")


@main.route("/fill_db")
def fill_database():
    controller = MongoController(mongo)
    face_det = FaceDetectionLibWrapper("./nn_inference/configs/test_fr_hog_config.json")
    FillDatabase(controller, face_det)("./face_database")
    return "Workers updated"


# @app.route('/objects', methods=['GET'])
# def object_detection_pipeline():
#     image = PLACEHOLDER_read_main_cam(0)
#     res = PLACEHOLDER_detect_objects(image)
#     form = None
#     return render_template("PLACEHOLDER_objects.html", form=form)
