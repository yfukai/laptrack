from typing import Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from ._typing_utils import IntArray
from ._typing_utils import Matrix


def lap_optimization(cost_matrix: Matrix) -> Tuple[IntArray, IntArray]:
    """Solves the linear assignment problem for a sparse matrix.

    Parameters
    ----------
    cost_matrix : lil_matrix or coo_matrix
        the cost matrix in scipy.sparse.lil_matrix format

    Returns
    -------
    xs : IntArray
        the indices such that assigned indices are (i,x[i]) (i=0 ... )
    ys : IntArray
        the indices such that assigned indices are (y[j],j) (j=0 ... )
    """
    rows, cols = min_weight_full_bipartite_matching(coo_matrix(cost_matrix))
    xs = cols[np.argsort(rows)]
    ys = rows[np.argsort(cols)]
    return xs, ys
