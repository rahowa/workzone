from os import stat
import typer
from typing import Sequence, Union, List, Optional
from app.nn_inference.common.base_wrapper import BaseWrapper

from app.base_types import Image
from app.result_types import BaseResult

class NetworkContext:
    def __init__(self, tag: str = "Default context") -> None:
        self.tag = tag
        self._model: Optional[BaseWrapper] = None

    def __repr__(self):
        return f"Context: [{self.tag}]\nmodel: [{self._model}]"

    @property
    def model(self) -> Optional[BaseWrapper]:
        return self._model

    @model.getter
    def model(self) -> Optional[BaseWrapper]:
        return self._model

    @model.setter
    def model(self, new_model: BaseWrapper) -> None:
        self._model.unload()
        self._model = new_model
        status = self._model.load()
        if not status:
            raise ValueError

    def predict(self, image: Union[Image, Sequence[Image]]) -> List[BaseResult]:
        if self._model is not None:
            return self._model.predict(image)
        else:
            print("Model was not assigned to context")
            exit(-1)
