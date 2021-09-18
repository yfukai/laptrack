"""Main module for tracking."""
from typing import Callable
from typing import Sequence
from typing import Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import lil_matrix

from ._cost_matrix import build_frame_cost_matrix, build_segment_cost_matrix
from ._optimization import lap_optimization
from ._typing_utils import Float
from ._typing_utils import FloatArray
from ._typing_utils import Int


def laptrack(
    coords: Sequence[FloatArray],
    track_cost_cutoff: Float = 15 ** 2,
    track_start_cost: Float = 30,  # b in Jaqaman et al 2008 NMeth.
    track_end_cost: Float = 30,  # d in Jaqaman et al 2008 NMeth.
    gap_closing_cost_cutoff: Union[Float, Literal[False]] = 15 ** 2,
    gap_closing_max_frame_count: Int = 2,
    splitting_cost_cutoff: Union[Float, Literal[False]] = False,
    no_splitting_cost: Float = 30,  # d' in Jaqaman et al 2008 NMeth.
    merging_cost_cutoff: Union[Float, Literal[False]] = False,
    no_merging_cost: Float = 30,  # b' in Jaqaman et al 2008 NMeth.
    dist_metric: Union[str, Callable] = "sqeuclidean",
) -> nx.Graph:
    """Track points by solving linear assignment problem.

    Parameters
    ----------
    coords : Sequence[FloatArray]
        The list of coordinates of point for each frame.
        The array index means (sample, dimension).
    track_cost_cutoff : Float, optional
        The cost cutoff for the connected points in the track.
        For default cases with `dist_metric="sqeuclidean"`,
        this value should be squared maximum distance.
        By default 15**2.
    track_start_cost : Float, optional
        The cost for starting the track (b in Jaqaman et al 2008 NMeth), by default 30
    track_end_cost : Float, optional
        The cost for ending the track (d in Jaqaman et al 2008 NMeth), by default 30
    gap_closing_cost_cutoff : Union[Float,Literal[False]] = 15,
        The cost cutoff for gap closing.
        For default cases with `dist_metric="sqeuclidean"`,
        this value should be squared maximum distance.
        If False, no splitting is allowed.
        By default 15**2.
    gap_closing_max_frame_count : Int = 2,
        The maximum frame gaps, by default 2.
    splitting_cost_cutoff : Union[Float,Literal[False]], optional
        The cost cutoff for the splitting points.
        For default cases with `dist_metric="sqeuclidean"`,
        this value should be squared maximum distance.
        If False, no splitting is allowed.
        By default False.
    no_splitting_cost : Float, optional
        The cost to reject splitting, by default 30.
    merging_cost_cutoff : Union[Float,Literal[False]], optional
        The cost cutoff for the merging points.
        For default cases with `dist_metric="sqeuclidean"`,
        this value should be squared maximum distance.
        If False, no merging is allowed.
        By default False.
    no_merging_cost : Float, optional
        The cost to reject merging, by default 30.
    dist_metric : Union[str, Callable], optional
        The metric for calculating cost,
        by default 'sqeuclidean' (squared euclidean distance).
        See documentation for `scipy.spatial.distance.cdist` for accepted values.

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
        dist_matrix = cdist(coord1, coord2, metric=dist_metric)
        cost_matrix = build_frame_cost_matrix(
            dist_matrix,
            track_cost_cutoff=track_cost_cutoff,
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

    if gap_closing_cost_cutoff or splitting_cost_cutoff or merging_cost_cutoff:
        # linking between tracks
        segments = list(nx.connected_components(track_tree))
        N_segments = len(segments)
        first_nodes = np.array(
            list(map(lambda segment: min(segment, key=lambda node: node[0]), segments))
        )
        last_nodes = np.array(
            list(map(lambda segment: max(segment, key=lambda node: node[0]), segments))
        )
        segments_df = pd.DataFrame(
            {
                "segment": segments,
                "first_frame": first_nodes[:, 0],
                "first_index": first_nodes[:, 1],
                "last_frame": first_nodes[:, 0],
                "last_index": first_nodes[:, 1],
            }
        ).reset_index()

        for prefix in ["first", "last"]:
            segments_df[f"{prefix}_frame_coords"] = segments_df.apply(
                lambda row: coords[row[f"{prefix}_frame"]][row[f"{prefix}_index"]],
                axis=1,
            )

        # compute candidate for gap closing
        if gap_closing_cost_cutoff:

            def to_gap_closing_candidates(row):
                target_coord = row["last_frame_coords"]
                indices = (
                    np.abs(segments_df["first_frame"] - row["last_frame"])
                    < gap_closing_max_frame_count
                )
                df = segments_df[indices]
                # note: can use KDTree if metric is distance,
                # but might not be appropriate for general metrics
                # https://stackoverflow.com/questions/35459306/find-points-within-cutoff-distance-of-other-points-with-scipy # noqa
                target_dist_matrix = cdist(
                    [target_coord],
                    np.stack(df["first_frame_coords"].values),
                    metric=dist_metric,
                )
                assert target_dist_matrix.shape[0] == 1
                indices2 = np.where(target_dist_matrix[0] < gap_closing_cost_cutoff)[0]
                return indices[indices2], target_dist_matrix[0][indices2]

            segments_df["gap_closing_candidates"] = segments_df.apply(
                to_gap_closing_candidates, axis=1
            )
        else:
            segments_df["gap_closing_candidates"] = [[]] * len(segments_df)

        gap_closing_dist_matrix = lil_matrix((N_segments, N_segments), dtype=np.float32)
        for ind, row in segments_df.iterrows():
            candidate_inds = row["gap_closing_candidates"][0]
            candidate_costs = row["gap_closing_candidates"][1]
            # row ... track end, col ... track start
            gap_closing_dist_matrix[(ind, candidate_inds)] = candidate_costs

        all_candidates = {}
        dist_matrices = {}

        # compute candidate for splitting and merging
        for prefix, cutoff in zip(
            ["first", "last"], [splitting_cost_cutoff, merging_cost_cutoff]
        ):
            if cutoff:

                def to_candidates(row):
                    target_coord = row[f"{prefix}_frame_coords"]
                    frame = row[f"{prefix}_frame"]
                    # note: can use KDTree if metric is distance,
                    # but might not be appropriate for general metrics
                    # https://stackoverflow.com/questions/35459306/find-points-within-cutoff-distance-of-other-points-with-scipy # noqa
                    target_dist_matrix = cdist(
                        [target_coord], coords[frame], metric=dist_metric
                    )
                    assert target_dist_matrix.shape[0] == 1
                    indices = np.where(target_dist_matrix[0] < cutoff)[0]
                    return [(frame, index) for index in indices], target_dist_matrix[0][
                        indices
                    ]

                segments_df[f"{prefix}_candidates"] = segments_df.apply(
                    to_candidates, axis=1
                )
            else:
                segments_df[f"{prefix}_candidates"] = [([], [])] * len(segments_df)

            all_candidates[prefix] = np.unique(
                np.concatenate(
                    segments_df[f"{prefix}_candidates"].apply(lambda x: x[0])
                ),
                axis=0,
            )

            N_middle = len(all_candidates[prefix])
            dist_matrices[prefix] = lil_matrix((N_segments, N_middle), dtype=np.float32)

            all_candidates_dict = {
                tuple(val): i for i, val in enumerate(all_candidates[prefix])
            }
            for ind, row in segments_df.iterrows():
                candidate_frame_indices = row[f"{prefix}_candidates"][0]
                candidate_inds = [
                    all_candidates_dict[tuple(fi)] for fi in candidate_frame_indices
                ]
                candidate_costs = row[f"{prefix}_candidates"][1]
                dist_matrices[prefix][(ind, candidate_inds)] = candidate_costs

        splitting_dist_matrix = dist_matrices["first"]
        merging_dist_matrix = dist_matrices["last"]
        splitting_all_candidates = all_candidates["first"]
        merging_all_candidates = all_candidates["first"]
        cost_matrix = build_segment_cost_matrix(
            gap_closing_dist_matrix,
            splitting_dist_matrix,
            merging_dist_matrix,
            no_splitting_cost,
            no_merging_cost,
        )
    #    _, xs, _ = lap_optimization(cost_matrix)

    #        count1 = dist_matrix.shape[0]
    #        count2 = dist_matrix.shape[1]
    #        connections = [(i, xs[i]) for i in range(count1) if xs[i] < count2]
    #        # track_start=[i for i in range(count1) if xs[i]>count2]
    #        # track_end=[i for i in range(count2) if ys[i]>count1]
    #        for connection in connections:
    #            track_tree.add_edge((frame, connection[0]), (frame + 1, connection[1]))

    return track_tree


# %%
