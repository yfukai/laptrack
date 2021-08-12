"""Main module for tracking."""
from typing import Optional
from typing import Sequence
from typing import Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import networkx as nx
from scipy.spatial import distance_matrix

from ._cost_matrix import build_frame_cost_matrix
from ._optimization import lap_optimization
from ._typing_utils import Float
from ._typing_utils import FloatArray
from ._typing_utils import Int


def track(
    coords: Sequence[FloatArray],
    props: Optional[Sequence[FloatArray]] = None,
    track_distance_cutoff: Float = 15,
    track_start_cost: Float = 30,  # b in Jaqaman et al 2008 NMeth.
    track_end_cost: Float = 30,  # d in Jaqaman et al 2008 NMeth.
    gap_closing_cutoff: Union[Float, Literal[False]] = 15,
    gap_closing_max_frame_count: Int = 2,
    splitting_cutoff: Union[Float, Literal[False]] = False,
    no_splitting_cost: Float = 30,  # d' in Jaqaman et al 2008 NMeth.
    merging_cutoff: Union[Float, Literal[False]] = False,
    no_merging_cost: Float = 30,  # b' in Jaqaman et al 2008 NMeth.
) -> nx.Graph:
    """Track points by solving linear assignment problem.

    Parameters
    ----------
    coords : Sequence[FloatArray]
        The list of coordinates of point for each frame.
        The array index means (sample, dimension).
    props : Optional[Sequence[FloatArray]], optional
        The properties (such as intensity) of the points (optional), by default None
    track_distance_cutoff : Float, optional
        The distance cutoff for the connected points in the track, by default 15
    track_start_cost : Float, optional
        The cost for starting the track (b in Jaqaman et al 2008 NMeth), by default 30
    track_end_cost : Float, optional
        The cost for ending the track (d in Jaqaman et al 2008 NMeth), by default 30
    gap_closing_cutoff : Union[Float,Literal[False]] = 15,
        The distance cutoff for gap closing, by default 15.
        If False, no splitting is allowed.
    gap_closing_max_frame_count : Int = 2,
        The maximum frame gaps, by default 2.
    splitting_cutoff : Union[Float,Literal[False]], optional
        The distance cutoff for the splitting points, by default 15.
        If False, no splitting is allowed.
    no_splitting_cost : Float, optional
        The cost to reject splitting, by default 30.
    merging_cutoff : Union[Float,Literal[False]], optional
        The distance cutoff for the merging points, by default 15.
        If False, no merging is allowed.
    no_merging_cost : Float, optional
        The cost to reject merging, by default 30.

    Returns
    -------
    tracks networkx.Graph:
        The graph for the tracks, whose nodes are (frame, index).
    """
    if any(list(map(lambda coord: coord.ndim != 2, coords))):
        raise ValueError("the elements in coords must be 2-dim.")
    coord_dim = coords[0].shape[1]
    if any(list(map(lambda coord: coord.shape[1] != coord_dim, coords))):
        raise ValueError("the second dimension in coords must have the same size")
    if props:
        if len(coords) != len(props):
            raise ValueError("the coords and props must have the same length.")
        if any(
            list(
                map(
                    lambda coord_prop: coord_prop[0].shape[0] != coord_prop[1].shape[0],
                    zip(coords, props),
                )
            )
        ):
            raise ValueError(
                "the number of coords and props must be the same for each frame."
            )

    # initialize tree
    track_tree = nx.Graph()
    for frame, coord in enumerate(coords):
        for j in range(coord.shape[0]):
            track_tree.add_node((frame, j))

    # linking between frames
    for frame, (coord1, coord2) in enumerate(zip(coords[:-1], coords[1:])):
        dist_matrix = distance_matrix(coord1, coord2)
        cost_matrix = build_frame_cost_matrix(
            dist_matrix,
            track_distance_cutoff=track_distance_cutoff,
            track_start_cost=track_start_cost,
            track_end_cost=track_end_cost,
        )
        _, xs, _ = lap_optimization(cost_matrix)

        count1 = dist_matrix.shape[0]
        count2 = dist_matrix.shape[1]
        connections = [(i, xs[i]) for i in range(count1) if xs[i] < count2]
        # track_start=[i for i in range(count1) if xs[i]>count2]
        # track_end=[i for i in range(count2) if ys[i]>count1]
        for connection in connections:
            track_tree.add_edge((frame, connection[0]), (frame + 1, connection[1]))

    #    if gap_closing_cutoff or splitting_cutoff or merging_cutoff:
    #        # linking between tracks
    #        segments = nx.connected_components(track_tree)

    return track_tree
