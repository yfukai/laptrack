from typing import Tuple

import lap
from scipy.sparse import lil_matrix
from scipy.sparse.csr import csr_matrix

from ._typing_utils import FloatArray
from ._typing_utils import Int
from ._typing_utils import IntArray


def __to_lap_sparse(
    cost_matrix: lil_matrix,
) -> Tuple[Int, FloatArray, IntArray, IntArray]:
    """Convert data for lap.lapmod."""
    n = cost_matrix.shape[0]
    cost_matrix2 = csr_matrix(cost_matrix)
    assert cost_matrix2.has_sorted_indices
    return n, cost_matrix2.data, cost_matrix2.indptr, cost_matrix2.indices


def lap_optimization(cost_matrix: lil_matrix) -> Tuple[float, IntArray, IntArray]:
    """Solves the linear assignment problem for a sparse matrix.

    Parameters
    ----------
    cost_matrix : lil_matrix
        the cost matrix in scipy.sparse.lil_matrix format

    Returns
    -------
    cost : float
        the final estimated cost
    xs : IntArray
        the indices such that assigned indices are (i,x[i]) (i=0 ... )
    ys : IntArray
        the indices such that assigned indices are (y[j],j) (j=0 ... )
    """
    return lap.lapmod(*__to_lap_sparse(cost_matrix))
