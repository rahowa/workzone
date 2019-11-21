from typing import List
from dataclasses import dataclass 

import cv2
import werkzeug
import numpy as np 
import jsonpickle
from flask import Flask, request, render_template, Response

from app import app
from TEST_face_detection import BBox, BBoxes


@dataclass
class DetectionResult:
    image: np.ndarray 
    labels: List[str]
    bboxes: BBoxes


            
def PLACEHOLDER_detect_objects(image: Image) -> DetectionResult:
    """
    """
    return DetectionResult(image, "human", ((20, 20, 50, 50)))

def PLACEHOLDER_define_workzone(image: Image) -> Image:
    """
    """
    pass

def PLACEHOLDER_read_main_cam(cam_id: int) -> Image:
    """
    """
    pass 

def PLACEHOLDER_recognaize_face(image: Image) -> DetectionResult:
    """
    """
    return DetectionResult(image, "Ilon mosk", ((20, 20, 50, 50)))


# @app.route('/workzone', methods=['GET'])
# def working_zone_pipeline():
#     image = PLACEHOLDER_read_main_cam(0)
#     image = PLACEHOLDER_define_workzone(image)
#     form = None
#     return render_template('PLACEHOLDER_workzone.html', form=form)


@app.route('/face', methods=['GET', 'POST'])
def face_recognition_pipeline():
    image = np.fromstring(request.data, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    detection_result = PLACEHOLDER_recognaize_face(image)
    response = {'message': f"{detection_result.labels} was detected"}
    response = jsonpickle.encode(response)
    return Response(response, status=200, mimetype="application/json")


# @app.route('/objects', methods=['GET'])
# def object_detection_pipeline():
#     image = PLACEHOLDER_read_main_cam(0)
#     res = PLACEHOLDER_detect_objects(image)
#     form = None
#     return render_template("PLACEHOLDER_objects.html", form=form)

