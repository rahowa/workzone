from typing import List

from app.base_types import Descriptors
from app.extensions import cache
from app.base_types import Image, Boxes
from app.mongo_controller import MongoController
from app.nn_inference.faces.wrappers.face_recognition_lib_wrapper import FaceRecognitionLibWrapper


@cache.cached(3000)
def get_known_encodings(db_controller: MongoController) -> Descriptors:
    return db_controller.all_valid_descriptors()


def check_persons(frame: Image, db_controller: MongoController) -> List[int]:
    known_face_encodings = get_known_encodings(db_controller)
    if len(known_face_encodings) == 0:
        return list()

    face_det = FaceRecognitionLibWrapper({"model_type": "hog", "number_of_times_to_upsample": 1})
    face_encodings = face_det.predict(frame).descriptors
    labels = list()
    for face_encoding in face_encodings:
        label = -1
        matches = face_det.match(known_face_encodings, face_encoding)
        if True in matches:
            label = matches.index(True)
        labels.append(label)
    return labels


def detect_faces(frame: Image) -> Boxes:
    face_det = FaceRecognitionLibWrapper({"model_type": "hog", "number_of_times_to_upsample": 1})
    face_locations = face_det.get_locations(frame)
    return face_locations
