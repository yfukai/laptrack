from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix

Int = Union[int, np.int_, np.uint8, np.uint16, np.uint32, np.uint64]
Float = Union[float, np.float64, np.float32, np.float16]
Ntype = TypeVar("Ntype", bound=np.number)
Itype = TypeVar("Itype", bound=np.integer)
Ftype = TypeVar("Ftype", bound=np.floating)

NumArray = npt.NDArray[Ntype]
FloatArray = npt.NDArray[Ftype]
IntArray = npt.NDArray[Itype]


Matrix = Union[FloatArray, coo_matrix, lil_matrix]
EdgeType = Union[
    nx.classes.reportviews.EdgeView, Sequence[Tuple[Tuple[Int, Int], Tuple[Int, Int]]]
]
