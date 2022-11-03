from os import PathLike as os_PathLike
from typing import Union
import numpy as np
from numpy.typing import NDArray

ArrayLike = Union[float, NDArray[np.float64]]
PathLike = Union[os_PathLike, str]
