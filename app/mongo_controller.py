import sys 
from collections import deque
from typing import Dict, Any, Union, List
from flask_pymongo import PyMongo

sys.path.append('../')
from nn_inference.base_wrapper import Descriptor

class MongoController:
    def __init__(self, model: PyMongo, table: str=None):
        self.model = model
        if table is not None:
            self.table = getattr(model.db, table)
        else: 
            self.table = model.db.workers

    def add_worker(self, worker: Dict[str, List[Dict[str, Any]]]) -> bool:
        before = self.table.count_documents()
        self.table.insert_one(worker)
        after = self.table.count_documents()
        if after > before:
            return True
        else:
            return False

    def remove_worker(self, worker: Any) -> bool:
        before = self.table.count_documents()
        self.table.delete_one(worker)
        after = self.table.count_documents()
        if after < before:
            return True
        else:
            return False

    def find_worker(self, worker_id: Union[Descriptor, str]) -> Dict[str, Descriptor]:
        if isinstance(woreker_id, (str)):
            return self.table.find({'id': worker_id})
        elif isinstance(worker_id, (Descriptor)):
            return self.table.find({"face_descriptor": worker_id})
        else:
            raise AttributeError 
    
    def all_descriptors(condition: Dict[str, Any]=None) -> List[Descriptor]:
        if condition is not None:
            workers = self.table.find(condition)
        else:
            workers = self.table.find()

        descriptors = deque(maxlen=len(workers))
        for worker in workers:
            descriptors.append(worker['face_descriptor'])
        return descriptors

    def update_descriptor(self, worker: Any, descriptor: Descriptor) ->bool:
        pass 