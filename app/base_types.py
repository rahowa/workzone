from typing import Tuple
from nptyping import Array


Image = Array[int]
Descriptor = Array[float]
Descriptors = Tuple[Array[float], ...]
BBox = Tuple[int, int, int, int]
BBoxes = Tuple[BBox, ...]