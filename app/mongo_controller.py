import sys 
from typing import Dict, Any, Union
from flask_pymongo import PyMongo

sys.path.append('../')
from nn_inference.base_wrapper import Descriptor

class MongoController:
    def __init__(self, model: PyMongo):
        self.model = model
    
    def check_worker(self, descriptor: Descriptor) -> bool:
        pass 

    def add_worker(self, worker: Dict[str, Descriptor]) -> bool:
        pass 
    
    def remove_worker(self, worker: Any) -> bool:
        pass 

    def find_worker(self, id: Union[Descriptor, str]) -> Dict[str, Descriptor]:
        pass 
    
    