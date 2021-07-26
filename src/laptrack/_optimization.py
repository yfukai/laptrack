from typing import Tuple
from ._typing_utils import IntArray

def lap_optimization(rows : IntArray, cols : IntArray, vals : IntArray) -> Tuple[IntArray,IntArray]:
    """Solves the linear assignment problem of sparse matrix.

    Parameters
    ----------
    rows : IntArray
        the row indices of the cost matrix.
    cols : IntArray
        the column indices of the cost matrix.
    vals : IntArray
        the values of the cost matrix.
    """
    pass