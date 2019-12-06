from typing import List, Union
from collections import deque
from dataclasses import dataclass 

import cv2
import werkzeug
import numpy as np 
import jsonpickle
from flask import Flask, request, render_template, Response
import face_recognition as fr 

from app import app
from TEST_face_detection import BBox, BBoxes
from .camera import Image
from .extensions import mongo, PyMongo
from .mongo_controller import MongoController
from nn_inference.external_wrappers import FaceDetectionLibWrapper

@dataclass
class DetectionResult:
    image: np.ndarray 
    labels: List[Union[int, str]]
    bboxes: BBoxes


def PLACEHOLDER_detect_objects(image: Image) -> DetectionResult:
    """
    """
    return DetectionResult(image, "human", ((20, 20, 50, 50)))


def PLACEHOLDER_recognaize_face(image: Image, db_controller: MongoController) -> DetectionResult:
    """
    """
    face_locations = fr.face_locations(image)
    face_encodings = fr.face_encodings(image, face_locations)
    known_face_encodings = db_controller.all_descriptors()
    labels = deque(maxlen=len(face_encodings))

    for face_encoding in face_encodings:
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        label = -1
        if True in matches:
            label = matches.index(True)
        labels.append(label)
    return DetectionResult(image, labels, face_locations)


# @app.route('/workzone', methods=['GET'])
# def working_zone_pipeline():
#     image = PLACEHOLDER_read_main_cam(0)
#     image = PLACEHOLDER_define_workzone(image)
#     form = None
#     return render_template('PLACEHOLDER_workzone.html', form=form)


@app.route('/face', methods=['GET', 'POST'])
def face_recognition_pipeline():
    if request.method == 'POST':
        image = np.fromstring(request.data, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        detection_result = PLACEHOLDER_recognaize_face(image)
        response = {'message': f"{detection_result.labels} was detected"}
        response = jsonpickle.encode(response)
        return Response(response, status=200, mimetype="application/json")
    else:
        return render_template("session_history.html")


@app.route('/objects', methods=['GET'])
def object_detection_pipeline():
    image = PLACEHOLDER_read_main_cam(0)
    res = PLACEHOLDER_detect_objects(image)
    form = None
    return render_template("PLACEHOLDER_objects.html", form=form)

