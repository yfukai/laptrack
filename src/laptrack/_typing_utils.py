from typing import Sequence
from typing import Tuple
from typing import Union

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix

NumArray = npt.NDArray[Union[np.float_, np.int_]]
FloatArray = npt.NDArray[np.float_]
IntArray = npt.NDArray[np.int_]

Int = Union[int, np.int_, np.uint8, np.uint16, np.uint32, np.uint64]
Float = Union[float, np.float_]

Matrix = Union[FloatArray, coo_matrix, lil_matrix]
EdgeType = Union[
    nx.classes.reportviews.EdgeView, Sequence[Tuple[Tuple[Int, Int], Tuple[Int, Int]]]
]
