import numpy as np
from typing import Tuple
from nptyping import Array


Image = Array[np.uint8]
Descriptor = Array[float]
Descriptors = Tuple[Array[float], ...]
Box = Tuple[int, int, int, int]
Boxes = Tuple[Box, ...]

