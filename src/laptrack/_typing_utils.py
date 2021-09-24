from typing import Any
from typing import Union

import numpy as np
import numpy.typing as npt
from scipy.sparse import lil_matrix
from scipy.sparse.coo import coo_matrix

NumArray = npt.NDArray[Any]
FloatArray = npt.NDArray[np.float_]
IntArray = npt.NDArray[np.int_]

Int = Union[int, np.int_]
Float = Union[float, np.float_]

Matrix = Union[FloatArray, coo_matrix, lil_matrix]
