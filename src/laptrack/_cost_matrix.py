from typing import Optional
from typing import Union

import numpy as np
from scipy.sparse import lil_matrix

from ._typing_utils import Float
from ._typing_utils import FloatArray


def build_frame_cost_matrix(
    dist_matrix: FloatArray,
    track_cost_cutoff: Float,
    track_start_cost: Optional[Float],
    track_end_cost: Optional[Float],
) -> lil_matrix:
    """Build sparce array for frame-linking cost matrix.

    Parameters
    ----------
    dist_matrix : FloatArray
        The distance matrix for points at time t and t+1.
    track_cost_cutoff : Float, optional
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
    ind = np.where(dist_matrix < track_cost_cutoff)
    C[(*ind,)] = dist_matrix[(*ind,)]
    if track_start_cost is None:
        track_start_cost = np.concatenate(C.data).max() * 1.05
    if track_end_cost is None:
        track_end_cost = np.concatenate(C.data).max() * 1.05
    C[np.arange(M, M + N), np.arange(N)] = np.ones(N) * track_end_cost
    C[np.arange(M), np.arange(N, N + M)] = np.ones(M) * track_start_cost
    ind2 = [ind[1] + M, ind[0] + N]
    min_val = np.concatenate(C.data).min() if len(C.data) > 0 else 0
    C[(*ind2,)] = min_val

    return C


def build_segment_cost_matrix(
    gap_closing_dist_matrix: Union[lil_matrix, FloatArray],
    splitting_dist_matrix: Union[lil_matrix, FloatArray],
    merging_dist_matrix: Union[lil_matrix, FloatArray],
    track_start_cost: Optional[Float],
    track_end_cost: Optional[Float],
    no_splitting_cost: Optional[Float],
    no_merging_cost: Optional[Float],
    alternative_cost_factor: Float = 1.05,
    alternative_cost_percentile: Float = 90,
    alternative_cost_percentile_interpolation: str = "lower",
) -> lil_matrix:
    """Build sparce array for segment-linking cost matrix.

    Parameters
    ----------
    gap_closing_dist_matrix : lil_matrix or FloatArray
        The distance matrix for closing gaps between segment i and j.
    splitting_dist_matrix : lil_matrix or FloatArray
        The distance matrix for splitting between segment i and time/index j
    merging_dist_matrix : lil_matrix or FloatArray
        The distance matrix for merging between segment i and time/index j
    track_start_cost : Float, optional
        The cost for starting the track (b in Jaqaman et al 2008 NMeth)
    track_end_cost : Float, optional
        The cost for ending the track (d in Jaqaman et al 2008 NMeth)
    no_splitting_cost : Float, optional
        The cost to reject splitting (d' in Jaqaman et al 2008 NMeth)
    no_merging_cost : Float, optional
        The cost to reject merging (b' in Jaqaman et al 2008 NMeth)
    alternative_cost_factor: Float
        The factor to calculate the alternative costs, by default 1.05.
    alternative_cost_percentile: Float
        The percentile to calculate the alternative costs, by default 90.
    alternative_cost_percentile_interpolation: str
        The percentile interpolation to calculate the alternative costs,
        by default "lower".
        See `numpy.percentile` for allowed values.

    Returns
    -------
    cost_matrix : Optional[FloatArray]
        the cost matrix for frame linking, None if not appropriate
    """
    M = gap_closing_dist_matrix.shape[0]
    assert gap_closing_dist_matrix.shape[1] == M
    assert splitting_dist_matrix.shape[0] == M
    assert merging_dist_matrix.shape[0] == M
    N1 = splitting_dist_matrix.shape[1]
    N2 = merging_dist_matrix.shape[1]

    S = 2 * M + N1 + N2

    C = lil_matrix((S, S), dtype=np.float32)

    C[:M, :M] = gap_closing_dist_matrix
    C[M : M + N1, :M] = splitting_dist_matrix.T
    C[:M, M : M + N2] = merging_dist_matrix

    all_data = np.concatenate(C.data)
    if len(all_data) == 0:
        return None

    # Note:
    # Though the way of assigning track_start_cost, track_end_cost, no_splitting_cost, no_merging_cost # noqa :
    # and min_val is similar to that of TrackMate (link1, link2), GPL3 of TrackMate does not apply. (See link3 for license discussion.) # noqa :
    #   link1 https://github.com/fiji/TrackMate/blob/5a97426586b3c592c986c57aa1a09bab9d21419c/src/main/java/fiji/plugin/trackmate/tracking/sparselap/costmatrix/DefaultCostMatrixCreator.java#L186 # noqa :
    #         https://github.com/fiji/TrackMate/blob/5a97426586b3c592c986c57aa1a09bab9d21419c/src/main/java/fiji/plugin/trackmate/tracking/sparselap/costmatrix/JaqamanSegmentCostMatrixCreator.java # noqa:
    #         https://github.com/fiji/TrackMate/blob/5a97426586b3c592c986c57aa1a09bab9d21419c/src/main/java/fiji/plugin/trackmate/tracking/sparselap/SparseLAPSegmentTracker.java#L148 # noqa:
    #   link2 (default parameters for alternative_cost_percentile, alternative_cost_factor) # noqa :
    #         https://github.com/fiji/TrackMate/blob/5a97426586b3c592c986c57aa1a09bab9d21419c/src/main/java/fiji/plugin/trackmate/tracking/TrackerKeys.java # noqa :
    #   link3 https://forum.image.sc/t/linear-assignment-problem-based-tracking-package-in-python/57793 # noqa :
    #         https://web.archive.org/web/20210921134401/https://forum.image.sc/t/linear-assignment-problem-based-tracking-package-in-python/57793 # noqa :

    if (
        track_start_cost is None
        or track_end_cost is None
        or no_splitting_cost is None
        or no_merging_cost is None
    ):
        alternative_cost = (
            np.percentile(
                all_data,
                alternative_cost_percentile,
                interpolation=alternative_cost_percentile_interpolation,
            )
            * alternative_cost_factor
        )
        if track_start_cost is None:
            track_start_cost = alternative_cost
        if track_end_cost is None:
            track_end_cost = alternative_cost
        if no_splitting_cost is None:
            no_splitting_cost = alternative_cost
        if no_merging_cost is None:
            no_merging_cost = alternative_cost

    C[np.arange(M + N1, 2 * M + N1), np.arange(M)] = np.ones(M) * track_start_cost
    C[np.arange(2 * M + N1, S), np.arange(M, M + N2)] = np.ones(N2) * no_merging_cost
    C[np.arange(M), np.arange(M + N2, 2 * M + N2)] = np.ones(M) * track_end_cost
    C[np.arange(M, M + N1), np.arange(2 * M + N2, S)] = np.ones(N1) * no_splitting_cost
    subC = C[: M + N1, : M + N2]
    inds = subC.nonzero()
    min_val = np.min(
        [track_start_cost, track_end_cost, no_splitting_cost, no_merging_cost]
    )
    C[inds[1] + M + N1, inds[0] + M + N2] = min_val

    return C
