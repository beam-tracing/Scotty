from os import PathLike as os_PathLike
from typing import Union
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ArrayLike = Union[float, FloatArray]
PathLike = Union[os_PathLike, str]
