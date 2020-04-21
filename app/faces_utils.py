from typing import List
from app.base_types import Image, Box, Boxes
from app.mongo_controller import MongoController
from app.nn_inference.faces.wrappers.face_recognition_lib_wrapper import FaceRecognitionLibWrapper


def check_persons(frame: Image, db_controller: MongoController) -> List[int]:
    known_face_encodings = db_controller.all_valid_descriptors()
    if len(known_face_encodings) == 0:
        return list()

    face_det = FaceRecognitionLibWrapper({"model_type": "hog", "number_of_times_to_upsample": 1})
    face_locations = face_det.get_locations(frame)
    face_encodings = face_det.get_encodings(frame, face_locations)
    labels = list()
    for face_encoding in face_encodings:
        label = -1
        matches = face_det.match(face_encoding, known_face_encodings)
        if True in matches:
            label = matches.index(True)
        labels.append(label)
    return labels


def detect_faces(frame: Image) -> Boxes:
    face_det = FaceRecognitionLibWrapper({"model_type": "hog", "number_of_times_to_upsample": 1})
    face_locations = face_det.get_locations(frame)
    return face_locations
