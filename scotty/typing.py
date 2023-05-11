# Copyright 2017 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

from os import PathLike as os_PathLike
from typing import Union
import numpy as np

try:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
except ImportError:
    FloatArray = np.ndarray  # type: ignore

ArrayLike = Union[float, FloatArray]
PathLike = Union[os_PathLike, str]
