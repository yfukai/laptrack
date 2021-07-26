import numpy as np
from scipy.sparse import lil_matrix

from ._typing_utils import Float
from ._typing_utils import FloatArray


def build_frame_cost_matrix(
    dist_matrix: FloatArray,
    track_distance_cutoff: Float,
    track_start_cost: Float,
    track_end_cost: Float,
) -> lil_matrix:
    """Build sparce array for frame-linking cost matrix.

    Parameters
    ----------
    dist_matrix : FloatArray
        The distance matrix for points at time t and t+1.
    track_distance_cutoff : Float, optional
        The distance cutoff for the connected points in the track
    track_start_cost : Float, optional
        The cost for starting the track (b in Jaqaman et al 2008 NMeth)
    track_end_cost : Float, optional
        The cost for ending the track (d in Jaqaman et al 2008 NMeth)

    Returns
    -------
    cost_matrix : FloatArray
        the cost matrix for frame linking
    """
    M = dist_matrix.shape[0]
    N = dist_matrix.shape[1]

    C = lil_matrix((M + N, N + M), dtype=np.float32)
    ind = np.where(dist_matrix < track_distance_cutoff)
    C[(*ind,)] = dist_matrix[(*ind,)]
    C[np.arange(M, M + N), np.arange(N)] = np.ones(N) * track_end_cost
    C[np.arange(M), np.arange(N, N + M)] = np.ones(M) * track_start_cost
    ind2 = [ind[1] + M, ind[0] + N]
    C[(*ind2,)] = np.min(dist_matrix)

    return C
