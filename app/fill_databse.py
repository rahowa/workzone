import os
import sys
import cv2
import numpy as np
from typing import List, Union
from collections import defaultdict, deque

from .mongo_controller import MongoController

sys.path.append('../')
from nn_inference.base_wrapper import BaseWrapper


class FillDatabase:

    def __init__(self, db_controller: MongoController,
                 fr_module: BaseWrapper):
        self.fr_module = fr_module
        self.persons_encodings = defaultdict(lambda: None)
        self.db_controller = db_controller

    def create_encodings(self, certain_persons: Union[str, List[str]]):
        if isinstance(certain_persons, (list, tuple)):
            path_to_faces = os.path.join(**certain_persons[0].split('/')[:-1])
            persons = certain_persons
        elif isinstance(certain_persons, (str)):
            path_to_faces = certain_persons
            persons = os.listdir(certain_persons)
        else:
            raise NotImplementedError

        for person in persons:
            path_to_person_faces = os.path.join(path_to_faces, person)
            persons_faces = os.listdir(path_to_person_faces)
            person_encoding_deq = deque(maxlen=len(persons_faces))

            for face_path in persons_faces:
                face_image = cv2.imread(os.path.join(path_to_person_faces, face_path))
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                face_image = self.fr_module.preprocess_image(face_image)
                face_bounding_boxes = self.fr_module.get_locations(face_image)

                if len(face_bounding_boxes) != 1:
                    print(f"{os.path.join(path_to_person_faces, face_path)} \
                         contains none or more than one faces and can't be used for verification.")
                    continue
                else:
                    face_enc = self.fr_module.get_encodings(face_image, face_bounding_boxes)
                    person_encoding_deq.append(face_enc)
            person_encoding_deq = np.asarray(person_encoding_deq)
            person_encoding = person_encoding_deq.mean(0).tolist()
            self.persons_encodings.update({person: person_encoding})

    def fill_database(self):
        for person, encoding in self.persons_encodings.items():
            if self.db_controller.add_worker({"name": person,
                                              "encoding": encoding}):
                print(f"Add new worker: {person}")
            else:
                print("Check databse settings")
                break

    def __call__(self, certain_encodings: Union[str, List[str]]):
        self.create_encodings(certain_encodings)
        self.fill_database()
