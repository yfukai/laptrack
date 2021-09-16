"""Main module for tracking."""
from typing import Sequence
from typing import Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import networkx as nx
import pandas as pd
from scipy.spatial import distance_matrix

from ._cost_matrix import build_frame_cost_matrix
from ._optimization import lap_optimization
from ._typing_utils import Float
from ._typing_utils import FloatArray
from ._typing_utils import Int


def track(
    coords: Sequence[FloatArray],
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

    if gap_closing_cutoff or splitting_cutoff or merging_cutoff:
        # linking between tracks
        segments = list(nx.connected_components(track_tree))
        first_nodes = map(
            lambda segment: min(segment, key=lambda node: node[0]), segments
        )
        last_nodes = map(
            lambda segment: max(segment, key=lambda node: node[0]), segments
        )
        segments_df = pd.DataFrame(
            {
                "segment": segments,
                "first_frame": list(map(lambda x: x[0], first_nodes)),
                "first_index": list(map(lambda x: x[1], first_nodes)),
                "last_frame": list(map(lambda x: x[0], last_nodes)),
                "last_index": list(map(lambda x: x[1], last_nodes)),
            }
        )

        for prefix in ["first", "last"]:
            segments_df[f"{prefix}_frame_coords"] = segments_df.apply(
                lambda row: coords[row[f"{prefix}_frame"]][row[f"{prefix}_index"]]
            )

        for prefix, cutoff in zip(
            ["first", "last"], [splitting_cutoff, merging_cutoff]
        ):
            for frame, grp in segments_df.groupby(f"{prefix}_frame"):
                target_coord = grp[f"{prefix}_frame_coords"]
                target_dist_matrix = distance_matrix(target_coord, coords[frame])
                is_candidate = target_dist_matrix < cutoff

    #        def get_candidates(frame,index,cutoff):
    #            coord=coords[frame][index]
    #        segments_df["split_candidates"] = seg#ments_df.apply(
    #            lambda row : row, axis=1         #
    #        )
    #        first_dist_matrix = distance_matrix(coord1, coord2)

    return track_tree
