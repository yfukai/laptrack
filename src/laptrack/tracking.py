from typing import Sequence, Optional, Union, List
from typing_extensions import Literal

import numpy as np
from numpy import typing as npt
from scipy.spatial import distance_matrix

from ._typing_utils import FloatArray, Float
from ._cost_matrix import build_frame_cost_matrix
from ._optimization import lap_optimization

def track_points(coords : Sequence[FloatArray], 
                 props : Optional[Sequence[FloatArray]] = None,
                 track_distance_cutoff : Float = 15,
                 track_start_cost : Float = 30, # b in Jaqaman et al 2008 NMeth.
                 track_end_cost : Float = 30, # d in Jaqaman et al 2008 NMeth.
                 gap_closing_cutoff : Union[Float,Literal[False]] = 15,
                 segment_splitting_cutoff : Union[Float,Literal[False]] = False,
                 no_splitting_cost : Float = 30, # d' in Jaqaman et al 2008 NMeth.
                 segment_merging_cutoff : Union[Float,Literal[False]] = False,
                 no_merging_cost : Float = 30, # b' in Jaqaman et al 2008 NMeth.
                 ) -> List[npt.NDArray[np.uint32]] :
    """Track points by solving linear assignment problem.

    Parameters
    ----------
    coords : Sequence[FloatArray]
        The list of coordinates of point for each frame. The array index means (sample, dimension). 
    props : Optional[Sequence[FloatArray]], optional
        The properties (such as intensity) of the points (optional), by default None
    track_distance_cutoff : Float, optional
        The distance cutoff for the connected points in the track, by default 15
    track_start_cost : Float, optional
        The cost for starting the track (b in Jaqaman et al 2008 NMeth), by default 30
    track_end_cost : Float, optional
        The cost for ending the track (d in Jaqaman et al 2008 NMeth), by default 30
    segment_splitting_cutoff : Union[Float,Literal[False]], optional
        The distance cutoff for the splitting points, by default 15. If False, no splitting is allowed.
    no_splitting_cost : Float, optional
        The cost to reject splitting, by default 30.
    segment_merging_cutoff : Union[Float,Literal[False]], optional
        The distance cutoff for the merging points, by default 15. If False, no merging is allowed.
    no_merging_cost : Float, optional
        The cost to reject merging, by default 30.

    Returns
    -------
    track_indices : List[npt.NDArray[np.uint32]]
        The indices of each tracks, in the same form as that of coords.
    """

    if any(list(map(lambda coord : coord.ndim!=2,coords))):
        raise ValueError("the elements in coords must be 2-dim.")
    coord_dim=coords[0][0].shape[1]
    if any(list(map(lambda coord : coord.shape[1]!=coord_dim,coords))):
        raise ValueError("the second dimension in coords must have the same size")
    if props:
        if len(coords) != len(props):
            raise ValueError("the coords and props must have the same length.")
        if any(list(map(lambda coord_prop : coord_prop[0].shape[0]!=coord_prop[1].shape[0],zip(coords,props)))):
            raise ValueError("the number of coords and props must be the same for each frame.")

    # first linking between frames 
    for coord1, coord2 in zip(coords[:-1],coords[1:]):
        dist_matrix = distance_matrix(coord1,coord2)
        cost_matrix = build_frame_cost_matrix(dist_matrix,
                        track_distance_cutoff=track_distance_cutoff,
                        track_start_cost=track_start_cost,
                        track_end_cost=track_end_cost)
        _,xs,ys = lap_optimization(cost_matrix)