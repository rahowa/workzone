import sys
import numpy as np
from collections import deque
from typing import Dict, Any, Union, List
from flask_pymongo import PyMongo

sys.path.append('../')
from app.base_types import Descriptor, Descriptors


class MongoController:
    """
    MongoDB controller for interaction with database

    Parameters
    -----------
        model: PyMongo
            Instanciated database model

        table: str, None
            Name of table with workers
    """

    def __init__(self, model: PyMongo, table: str = None):
        self.model = model
        if table is not None:
            self.table = getattr(model.db, table)
        else:
            self.table = model.db.workers

    def add_worker(self, worker: Dict[str, List[Dict[str, Any]]]) -> bool:
        """
        Add new worker not databse

        Parameters
        -----------
            worker: Dict[str, List[Dict[str, Any]]]
                Dict with username and additional info

        Returns
        --------
            result: bool
                True if adding was successful
        """
        before = self.table.count_documents({})
        self.table.insert_one(worker)
        after = self.table.count_documents({})
        if after > before:
            return True
        else:
            return False

    def remove_worker(self, worker: Any) -> bool:
        """
        Remove worker form databse by attribute

        Parameters
        ----------
            worker: Any
                workers attribute

        Returns
        -------
            result: bool
                True if removing was successful
        """

        before = self.table.count_documents({})
        self.table.delete_one(worker)
        after = self.table.count_documents({})
        if after < before:
            return True
        else:
            return False

    def find_worker(self, worker_id: Union[Descriptor, str]) -> Dict[str, Descriptor]:
        """
        Find user by descriptor or username

        Parameters
        ----------
            worker_id: Union[Descriptor, str]
                Worker descriptor or username
            
        Returns:
            result: Dict[str, Descriptor]
                Username with descriptor
        """

        if isinstance(worker_id, (str)):
            return self.table.find({'id': worker_id})
        elif isinstance(worker_id, (Descriptor)):
            return self.table.find({"encoding": worker_id})
        else:
            raise AttributeError

    def all_descriptors(self, condition: Dict[str, Any] = None) -> Descriptors:
        """
        Return all descriptors of all workers by condition

        Parameters
        ----------
            condition: Dict[str, Any], None
                Condition to query

        Returns
        -------
            descriptors: Descriptors
                All finded descriptors
        """

        if condition is not None:
            workers = self.table.find(condition)
        else:
            workers = self.table.find()

        descriptors = deque()
        for worker in workers:
            encoding = worker['encoding']
            encoding = np.array(encoding).reshape(-1, )
            descriptors.append(encoding)
        return tuple(descriptors)

    def all_valid_descriptors(self, condition: Dict[str, Any] = None) -> Descriptors:
        """
        Return all valid descriptors of all workers by condition

        Parameters
        ----------
            condition: Dict[str, Any], None
                Condition to query

        Returns
        -------
            descriptors: Descriptors
                All finded descriptors
        """

        descriptors = self.all_descriptors(condition)
        return tuple(filter(lambda x: not np.isnan(x).all(), descriptors))

    def clean(self) -> None:
        self.table.drop()

