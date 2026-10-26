# Copyright 2017 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

from os import PathLike as os_PathLike
from typing import Any, Union, Tuple, Protocol
import numpy as np

try:
    from numpy.typing import NDArray

    Array = NDArray[np.number]
    FloatArray = NDArray[np.float64]
    ComplexFloatArray = NDArray[np.complex128]
except ImportError:
    Array = np.ndarray
    FloatArray = np.ndarray
    ComplexFloatArray = np.ndarray

ArrayLike = Union[float, np.floating[Any], FloatArray, ComplexFloatArray]
ComplexArrayLike = Union[complex, np.complexfloating[Any, Any], FloatArray]

CoordinateLike = Union[FloatArray, Tuple[ArrayLike, ArrayLike, ArrayLike]]
PathLike = Union[os_PathLike, str]
