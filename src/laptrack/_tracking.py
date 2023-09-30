"""Main module for tracking."""
import logging
import warnings
from enum import Enum
from functools import partial
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

# https://stackoverflow.com/questions/59037244/mypy-incompatible-import-error-for-conditional-imports # noqa :
if TYPE_CHECKING:
    from typing import Literal  # type: ignore
else:
    try:
        from typing import Literal
    except ImportError:
        from typing_extensions import Literal

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pydantic import BaseModel, Field, Extra


from ._cost_matrix import build_frame_cost_matrix, build_segment_cost_matrix
from ._optimization import lap_optimization
from ._typing_utils import NumArray
from ._typing_utils import Int
from ._typing_utils import EdgeType
from ._coo_matrix_builder import coo_matrix_builder
from .data_conversion import (
    convert_dataframe_to_coords_frame_index,
    convert_tree_to_dataframe,
)

logger = logging.getLogger(__name__)


class ParallelBackend(str, Enum):
    """Parallelization strategy for computation."""

    serial = "serial"
    ray = "ray"


def _get_segment_df(coords, track_tree):
    """
    Create segment dataframe from track tree.

    Parameters
    ----------
    coords :
        coordinates
    track_tree : nx.Graph
        the track tree

    Returns
    -------
    pd.DataFrame
        the segment dataframe, with the columns "segment", "first_frame", "first_index",
        "first_frame_coords", "last_frame", "last_index", "last_frame_coords"

    """
    # linking between tracks
    segments = list(nx.connected_components(track_tree))
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
            "last_frame": last_nodes[:, 0],
            "last_index": last_nodes[:, 1],
        }
    ).reset_index()

    for prefix in ["first", "last"]:
        segments_df[f"{prefix}_frame_coords"] = segments_df.apply(
            lambda row: coords[row[f"{prefix}_frame"]][row[f"{prefix}_index"]],
            axis=1,
        )
    return segments_df


class LapTrack(BaseModel, extra=Extra.forbid):
    """Tracking class for LAP tracker with parameters."""

    track_dist_metric: Union[str, Callable] = Field(
        "sqeuclidean",
        description="The metric for calculating track linking cost. "
        + "See documentation for `scipy.spatial.distance.cdist` for accepted values.",
    )
    track_cost_cutoff: float = Field(
        15**2,
        description="The cost cutoff for the connected points in the track. "
        + "For default cases with `dist_metric='sqeuclidean'`, "
        + "this value should be squared maximum distance.",
    )
    gap_closing_dist_metric: Union[str, Callable] = Field(
        "sqeuclidean",
        description="The metric for calculating gap closing cost. "
        + "See documentation for `scipy.spatial.distance.cdist` for accepted values.",
    )
    gap_closing_cost_cutoff: Union[Literal[False], float] = Field(
        15**2,
        description="The cost cutoff for gap closing. "
        + "For default cases with `dist_metric='sqeuclidean'`, "
        + "this value should be squared maximum distance. "
        + "If False, no gap closing is allowed.",
    )
    gap_closing_max_frame_count: int = Field(
        2, description="The maximum frame gaps, by default 2."
    )

    splitting_dist_metric: Union[str, Callable] = Field(
        "sqeuclidean",
        description="The metric for calculating splitting cost. "
        + "See `track_dist_metric`.",
    )
    splitting_cost_cutoff: Union[Literal[False], float] = Field(
        False,
        description="The cost cutoff for splitting. "
        + "See `gap_closing_cost_cutoff`. "
        + "If False, no splitting is allowed.",
    )
    merging_dist_metric: Union[str, Callable] = Field(
        "sqeuclidean",
        description="The metric for calculating merging cost. "
        + "See `track_dist_metric`",
    )
    merging_cost_cutoff: Union[Literal[False], float] = Field(
        False,
        description="The cost cutoff for merging. "
        + "See `gap_closing_cost_cutoff`. "
        + "If False, no merging is allowed.",
    )

    track_start_cost: Optional[float] = Field(
        None,  # b in Jaqaman et al 2008 NMeth.
        description="The cost for starting the track (b in Jaqaman et al 2008 NMeth) "
        + "If None, automatically estimated.",
    )
    track_end_cost: Optional[float] = Field(
        None,  # b in Jaqaman et al 2008 NMeth.
        description="The cost for ending the track (d in Jaqaman et al 2008 NMeth). "
        + "If None, automatically estimated.",
    )
    segment_start_cost: Optional[float] = Field(
        None,  # b in Jaqaman et al 2008 NMeth for segment connection
        description="The cost for starting the segment (b in Jaqaman et al 2008 NMeth). "
        + "If None, automatically estimated.",
    )
    segment_end_cost: Optional[float] = Field(
        None,  # b in Jaqaman et al 2008 NMeth for segment connection
        description="The cost for ending the segment (d in Jaqaman et al 2008 NMeth). "
        + "If None, automatically estimated.",
    )
    no_splitting_cost: Optional[float] = Field(
        None,  # d' in Jaqaman et al 2008 NMeth.
        description="The cost to reject splitting, if None, automatically estimated.",
    )
    no_merging_cost: Optional[float] = Field(
        None,  # d' in Jaqaman et al 2008 NMeth.
        description="The cost to reject merging, if None, automatically estimated.",
    )

    alternative_cost_factor: float = Field(
        1.05,
        description="The factor to calculate the alternative costs "
        + "(b,d,b',d' in Jaqaman et al 2008 NMeth).",
    )
    alternative_cost_percentile: float = Field(
        90, description="The percentile to calculate the alternative costs."
    )
    alternative_cost_percentile_interpolation: str = Field(
        "lower",
        description="The percentile interpolation to calculate the alternative costs. "
        + "See `numpy.percentile` for accepted values.",
    )
    parallel_backend: ParallelBackend = Field(
        ParallelBackend.serial,
        description="The parallelization strategy. "
        + f"Must be one of {', '.join([ps.name for ps in ParallelBackend])}.",
        exclude=True,
    )

    def _predict_links(
        self, coords, segment_connected_edges, split_merge_edges
    ) -> nx.Graph:
        """
        Link particles between frames according to the cost function.

        Parameters
        ----------
            coords : List[np.ndarray]
                the input coordinates
            segment_connected_edges : EdgesType
                the connected edges list that will be connected in this step
            split_merge_edges : EdgesType
                the connected edges list that will be connected in split and merge step

        Returns
        -------
            nx.Graph: The resulted tree.

        """
        # initialize tree
        track_tree = nx.Graph()
        for frame, coord in enumerate(coords):
            for j in range(coord.shape[0]):
                track_tree.add_node((frame, j))

        # linking between frames

        edges_list = list(segment_connected_edges) + list(split_merge_edges)

        def _predict_link_single_frame(
            frame: int,
            coord1: np.ndarray,
            coord2: np.ndarray,
        ) -> List[EdgeType]:
            force_end_indices = [e[0][1] for e in edges_list if e[0][0] == frame]
            force_start_indices = [e[1][1] for e in edges_list if e[1][0] == frame + 1]
            dist_matrix = cdist(coord1, coord2, metric=self.track_dist_metric)
            dist_matrix[force_end_indices, :] = np.inf
            dist_matrix[:, force_start_indices] = np.inf

            ind = np.where(dist_matrix < self.track_cost_cutoff)
            dist_matrix = coo_matrix_builder(
                dist_matrix.shape,
                row=ind[0],
                col=ind[1],
                data=dist_matrix[(*ind,)],
                dtype=dist_matrix.dtype,
            )

            cost_matrix = build_frame_cost_matrix(
                dist_matrix,
                track_start_cost=self.track_start_cost,
                track_end_cost=self.track_end_cost,
            )
            xs, _ = lap_optimization(cost_matrix)

            count1 = dist_matrix.shape[0]
            count2 = dist_matrix.shape[1]
            connections = [(i, xs[i]) for i in range(count1) if xs[i] < count2]
            edges: List[EdgeType] = [
                ((frame, connection[0]), (frame + 1, connection[1]))
                for connection in connections
            ]
            return edges

        if self.parallel_backend == ParallelBackend.serial:
            all_edges = []
            for frame, (coord1, coord2) in enumerate(zip(coords[:-1], coords[1:])):
                edges = _predict_link_single_frame(frame, coord1, coord2)
                all_edges.extend(edges)
        elif self.parallel_backend == ParallelBackend.ray:
            try:
                import ray
            except ImportError:
                raise ImportError("Please install `ray` to use `ParallelBackend.ray`.")
            remote_func = ray.remote(_predict_link_single_frame)
            res = [
                remote_func.remote(frame, coord1, coord2)
                for frame, (coord1, coord2) in enumerate(zip(coords[:-1], coords[1:]))
            ]
            all_edges = sum(ray.get(res), [])
        else:
            raise ValueError(
                f"Unknown parallel backend {self.parallel_backend}. "
                + f"Must be one of {', '.join([ps.name for ps in ParallelBackend])}."
            )

        track_tree.add_edges_from(all_edges)
        track_tree.add_edges_from(segment_connected_edges)
        return track_tree

    def _get_gap_closing_matrix(
        self, segments_df, *, force_end_nodes=[], force_start_nodes=[]
    ):
        """
        Generate the cost matrix for connecting segment ends.

        Parameters
        ----------
        segments_df : pd.DataFrame
            must have the columns "first_frame", "first_index", "first_crame_coords", "last_frame", "last_index", "last_frame_coords"
        force_end_nodes : list of int
            the indices of the segments_df that is forced to be end for future connection
        force_start_nodes : list of int
            the indices of the segments_df that is forced to be start for future connection

        Returns
        -------
        segments_df: pd.DataFrame
            the segments dataframe with additional column "gap_closing_candidates"
            (index of the candidate row of segments_df, the associated costs)
         gap_closing_dist_matrix: coo_matrix_builder
            the cost matrix for gap closing candidates

        """
        if self.gap_closing_cost_cutoff:

            def to_gap_closing_candidates(row, segments_df):
                # if the index is in force_end_indices, do not add to gap closing candidates
                if (row["last_frame"], row["last_index"]) in force_end_nodes:
                    return [], []

                target_coord = row["last_frame_coords"]
                frame_diff = segments_df["first_frame"] - row["last_frame"]

                # only take the elements that are within the frame difference range.
                # segments in df is later than the candidate segment (row)
                indices = (1 <= frame_diff) & (
                    frame_diff <= self.gap_closing_max_frame_count
                )
                df = segments_df[indices]
                force_start = df.apply(
                    lambda row: (row["first_frame"], row["first_index"])
                    in force_start_nodes,
                    axis=1,
                )
                df = df[~force_start]
                # do not connect to the segments that is forced to be start
                # note: can use KDTree if metric is distance,
                # but might not be appropriate for general metrics
                # https://stackoverflow.com/questions/35459306/find-points-within-cutoff-distance-of-other-points-with-scipy # noqa
                # TrackMate also uses this (trivial) implementation.
                if len(df) > 0:
                    target_dist_matrix = cdist(
                        [target_coord],
                        np.stack(df["first_frame_coords"].values),
                        metric=self.gap_closing_dist_metric,
                    )
                    assert target_dist_matrix.shape[0] == 1
                    indices2 = np.where(
                        target_dist_matrix[0] < self.gap_closing_cost_cutoff
                    )[0]
                    return (
                        df.index[indices2].values,
                        target_dist_matrix[0][indices2],
                    )
                else:
                    return [], []

            if self.parallel_backend == ParallelBackend.serial:
                segments_df["gap_closing_candidates"] = segments_df.apply(
                    partial(to_gap_closing_candidates, segments_df=segments_df), axis=1
                )
            elif self.parallel_backend == ParallelBackend.ray:
                try:
                    import ray
                except ImportError:
                    raise ImportError(
                        "Please install `ray` to use `ParallelBackend.ray`."
                    )
                remote_func = ray.remote(to_gap_closing_candidates)
                segments_df_id = ray.put(segments_df)
                res = [
                    remote_func.remote(row, segments_df_id)
                    for _, row in segments_df.iterrows()
                ]
                segments_df["gap_closing_candidates"] = ray.get(res)
            else:
                raise ValueError(f"Unknown parallel_backend {self.parallel_backend}. ")
        else:
            segments_df["gap_closing_candidates"] = [([], [])] * len(segments_df)

        N_segments = len(segments_df)
        gap_closing_dist_matrix = coo_matrix_builder(
            (N_segments, N_segments), dtype=np.float32
        )
        for ind, row in segments_df.iterrows():
            candidate_inds = row["gap_closing_candidates"][0]
            candidate_costs = row["gap_closing_candidates"][1]
            # row ... track end, col ... track start
            gap_closing_dist_matrix[
                (int(cast(int, ind)), candidate_inds)
            ] = candidate_costs

        return segments_df, gap_closing_dist_matrix

    def _get_splitting_merging_candidates(
        self,
        segments_df,
        coords,
        cutoff,
        prefix,
        dist_metric,
        *,
        force_end_nodes=[],
        force_start_nodes=[],
    ):
        if cutoff:

            def to_candidates(row, coords):
                # if the prefix is first, this means the row is the track start, and the target is the track end
                other_frame = row[f"{prefix}_frame"] + (-1 if prefix == "first" else 1)
                target_coord = row[f"{prefix}_frame_coords"]
                row_no_connection_nodes = (
                    force_start_nodes if prefix == "first" else force_end_nodes
                )
                other_no_connection_nodes = (
                    force_end_nodes if prefix == "first" else force_start_nodes
                )
                other_no_connection_indices = [
                    n[1] for n in other_no_connection_nodes if n[0] == other_frame
                ]

                if (
                    row[f"{prefix}_frame"],
                    row[f"{prefix}_index"],
                ) in row_no_connection_nodes:
                    return (
                        [],
                        [],
                    )  # do not connect to the segments that is forced to be start or end
                # note: can use KDTree if metric is distance,
                # but might not be appropriate for general metrics
                # https://stackoverflow.com/questions/35459306/find-points-within-cutoff-distance-of-other-points-with-scipy # noqa
                if other_frame < 0 or len(coords) <= other_frame:
                    return [], []
                target_dist_matrix = cdist(
                    [target_coord], coords[other_frame], metric=dist_metric
                )
                assert target_dist_matrix.shape[0] == 1
                target_dist_matrix[
                    0, other_no_connection_indices
                ] = (
                    np.inf
                )  # do not connect to the segments that is forced to be start or end
                indices = np.where(target_dist_matrix[0] < cutoff)[0]
                return [(other_frame, index) for index in indices], target_dist_matrix[
                    0
                ][indices]

            if self.parallel_backend == ParallelBackend.serial:
                segments_df[f"{prefix}_candidates"] = segments_df.apply(
                    partial(to_candidates, coords=coords), axis=1
                )
            elif self.parallel_backend == ParallelBackend.ray:
                try:
                    import ray
                except ImportError:
                    raise ImportError(
                        "Please install `ray` to use `ParallelBackend.ray`."
                    )
                remote_func = ray.remote(to_candidates)
                coords_id = ray.put(coords)
                res = [
                    remote_func.remote(row, coords_id)
                    for _, row in segments_df.iterrows()
                ]
                segments_df[f"{prefix}_candidates"] = ray.get(res)
            else:
                raise ValueError(f"Unknown parallel_backend {self.parallel_backend}. ")
        else:
            segments_df[f"{prefix}_candidates"] = [([], [])] * len(segments_df)

        middle_point_candidates = np.unique(
            sum(
                segments_df[f"{prefix}_candidates"].apply(lambda x: list(x[0])),
                [],
            ),
            axis=0,
        )

        N_segments = len(segments_df)
        N_middle = len(middle_point_candidates)
        dist_matrix = coo_matrix_builder((N_segments, N_middle), dtype=np.float32)

        middle_point_candidates_dict = {
            tuple(val): i for i, val in enumerate(middle_point_candidates)
        }
        for ind, row in segments_df.iterrows():
            candidate_frame_indices = row[f"{prefix}_candidates"][0]
            candidate_inds = [
                middle_point_candidates_dict[tuple(fi)]
                for fi in candidate_frame_indices
            ]
            candidate_costs = row[f"{prefix}_candidates"][1]
            dist_matrix[(int(cast(Int, ind)), candidate_inds)] = candidate_costs

        return segments_df, dist_matrix, middle_point_candidates

    def _link_gap_split_merge_from_matrix(
        self,
        segments_df,
        track_tree,
        gap_closing_dist_matrix,
        splitting_dist_matrix,
        merging_dist_matrix,
        splitting_all_candidates,
        merging_all_candidates,
    ):
        cost_matrix = build_segment_cost_matrix(
            gap_closing_dist_matrix,
            splitting_dist_matrix,
            merging_dist_matrix,
            self.segment_start_cost,
            self.segment_end_cost,
            self.no_splitting_cost,
            self.no_merging_cost,
            self.alternative_cost_factor,
            self.alternative_cost_percentile,
            self.alternative_cost_percentile_interpolation,
        )

        if not cost_matrix is None:
            # FIXME connected_edges_list

            xs, ys = lap_optimization(cost_matrix)

            M = gap_closing_dist_matrix.shape[0]
            N1 = splitting_dist_matrix.shape[1]
            N2 = merging_dist_matrix.shape[1]

            for ind, row in segments_df.iterrows():
                col_ind = xs[ind]
                first_frame_index = (row["first_frame"], row["first_index"])
                last_frame_index = (row["last_frame"], row["last_index"])
                if col_ind < M:
                    target_frame_index = tuple(
                        segments_df.loc[col_ind, ["first_frame", "first_index"]]
                    )
                    track_tree.add_edge(last_frame_index, target_frame_index)
                elif col_ind < M + N2:
                    track_tree.add_edge(
                        last_frame_index,
                        tuple(merging_all_candidates[col_ind - M]),
                    )

                row_ind = ys[ind]
                if M <= row_ind and row_ind < M + N1:
                    track_tree.add_edge(
                        first_frame_index,
                        tuple(splitting_all_candidates[row_ind - M]),
                    )

        return track_tree

    def _predict_gap_split_merge(self, coords, track_tree, split_edges, merge_edges):
        """
        Perform gap-closing and splitting/merging prediction.

        Parameters
        ----------
             coords : Sequence[NumArray]
                 The list of coordinates of point for each frame.
                 The array index means (sample, dimension).
             track_tree : nx.Graph
                 the track tree
             connected_edges_list (List[List[Tuple[Tuple[int, int],Tuple[int, int]]]]):
                 the connected edges list

        Returns
        -------
             track_tree : nx.Graph
                 the updated track tree
        """
        edges = list(split_edges) + list(merge_edges)
        if (
            self.gap_closing_cost_cutoff
            or self.splitting_cost_cutoff
            or self.merging_cost_cutoff
        ):
            segments_df = _get_segment_df(coords, track_tree)
            force_end_nodes = [tuple(map(int, e[0])) for e in edges]
            force_start_nodes = [tuple(map(int, e[1])) for e in edges]

            # compute candidate for gap closing
            segments_df, gap_closing_dist_matrix = self._get_gap_closing_matrix(
                segments_df,
                force_end_nodes=force_end_nodes,
                force_start_nodes=force_start_nodes,
            )

            middle_points: Dict = {}
            dist_matrices: Dict = {}

            # compute candidate for splitting and merging
            for prefix, cutoff, dist_metric in zip(
                ["first", "last"],
                [self.splitting_cost_cutoff, self.merging_cost_cutoff],
                [self.splitting_dist_metric, self.merging_dist_metric],
            ):
                (
                    segments_df,
                    dist_matrices[prefix],
                    middle_points[prefix],
                ) = self._get_splitting_merging_candidates(
                    segments_df,
                    coords,
                    cutoff,
                    prefix,
                    dist_metric,
                    force_end_nodes=force_end_nodes,
                    force_start_nodes=force_start_nodes,
                )

            splitting_dist_matrix = dist_matrices["first"]
            merging_dist_matrix = dist_matrices["last"]
            splitting_all_candidates = middle_points["first"]
            merging_all_candidates = middle_points["last"]

            track_tree = self._link_gap_split_merge_from_matrix(
                segments_df,
                track_tree,
                gap_closing_dist_matrix,
                splitting_dist_matrix,
                merging_dist_matrix,
                splitting_all_candidates,
                merging_all_candidates,
            )
        track_tree.add_edges_from(edges)
        return track_tree

    def predict(
        self,
        coords: Sequence[NumArray],
        connected_edges: Optional[EdgeType] = None,
        split_merge_validation: bool = True,
    ) -> nx.DiGraph:
        """Predict the tracking graph from coordinates.

        Parameters
        ----------
            coords : Sequence[NumArray]
                The list of coordinates of point for each frame.
                The array index means `(sample, dimension)`.
            connected_edges : Optional[EdgeType]
                The edges that is known to be connected.
                If None, no edges are assumed to be connected.
            split_merge_validation : bool
                If true, check if the split/merge edges are two-fold.


        Raises
        ------
            ValueError: Raised for invalid coordinate formats.

        Returns
        -------
            track_tree : nx.DiGraph
                The graph for the tracks, whose nodes are `(frame, index)`.
                The edge direction represents the time order.
        """
        if any(list(map(lambda coord: coord.ndim != 2, coords))):
            raise ValueError("the elements in coords must be 2-dim.")
        coord_dim = coords[0].shape[1]
        if any(list(map(lambda coord: coord.shape[1] != coord_dim, coords))):
            raise ValueError("the second dimension in coords must have the same size")

        if connected_edges:
            connected_edges = [
                (n1, n2) if n1[0] < n2[0] else (n2, n1) for n1, n2 in connected_edges
            ]
            tree = nx.from_edgelist(connected_edges)
            split_edges = []
            merge_edges = []
            for m in tree.nodes():
                successors = [n for n in tree.neighbors(m) if n[0] > m[0]]
                if len(successors) > 1:
                    if split_merge_validation:
                        assert len(successors) == 2, "splitting into >2 nodes"
                    split_edges.append((m, successors[0]))
                predecessors = [n for n in tree.neighbors(m) if n[0] < m[0]]
                if len(predecessors) > 1:
                    if split_merge_validation:
                        assert len(predecessors) == 2, "merging of >2 nodes"
                    merge_edges.append((predecessors[0], m))
            segment_connected_edges = list(
                set(connected_edges) - set(split_edges) - set(merge_edges)
            )
        else:
            segment_connected_edges = []
            split_edges = []
            merge_edges = []

        ####### Particle-particle tracking #######
        track_tree = self._predict_links(
            coords, segment_connected_edges, list(split_edges) + list(merge_edges)
        )
        track_tree = self._predict_gap_split_merge(
            coords, track_tree, split_edges, merge_edges
        )

        # convert to directed graph
        edges = [
            (n1, n2) if n1[0] < n2[0] else (n2, n1) for (n1, n2) in track_tree.edges()
        ]
        track_tree_directed = nx.DiGraph()
        track_tree_directed.add_nodes_from(track_tree.nodes())
        track_tree_directed.add_edges_from(edges)

        return track_tree_directed

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        coordinate_cols: List[str],
        frame_col: str = "frame",
        validate_frame: bool = True,
        only_coordinate_cols: bool = True,
        connected_edges: Optional[List[Tuple[Int, Int]]] = None,
        index_offset: Int = 0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Shorthand for the tracking with the dataframe input / output.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataframe.
        coordinate_cols : List[str]
            The list of the columns to use for coordinates.
        frame_col : str, optional
            The column name to use for the frame index. Defaults to "frame".
        validate_frame : bool, optional
            Whether to validate the frame. Defaults to True.
        only_coordinate_cols : bool, optional
            Whether to use only the coordinate columns. Defaults to True.
        connected_edges : Optional[List[Tuple[Int,Int]]]
            The edges that is known to be connected.
            Must be a list of the tuples of the row numbers (not index, but `iloc`).
            If None, no edges are assumed to be connected.
        index_offset : Int
            The offset to add to the track and tree index.

        Returns
        -------
        track_df : pd.DataFrame
            The track dataframe, with the following columns:

            - "frame" : The frame index.
            - "index" : The coordinate index.
            - "track_id" : The track id.
            - "tree_id" : The tree id.
            - the other columns : The coordinate values.
        split_df : pd.DataFrame
            The splitting dataframe, with the following columns:

            - "parent_track_id" : The track id of the parent.
            - "child_track_id" : The track id of the child.
        merge_df : pd.DataFrame
            The merging dataframe, with the following columns:

            - "parent_track_id" : The track id of the parent.
            - "child_track_id" : The track id of the child.
        """
        coords, frame_index = convert_dataframe_to_coords_frame_index(
            df, coordinate_cols, frame_col, validate_frame
        )
        if connected_edges is not None:
            connected_edges2 = [
                (frame_index[i1], frame_index[i2]) for i1, i2 in connected_edges
            ]
        else:
            connected_edges2 = None
        tree = self.predict(coords, connected_edges=connected_edges2)
        if only_coordinate_cols:
            track_df, split_df, merge_df = convert_tree_to_dataframe(tree, coords)
            track_df = track_df.rename(
                columns={f"coord-{i}": k for i, k in enumerate(coordinate_cols)}
            )
            warnings.warn(
                "The parameter only_coordinate_cols will be False by default in the major release.",
                FutureWarning,
            )
        else:
            track_df, split_df, merge_df = convert_tree_to_dataframe(
                tree, dataframe=df, frame_index=frame_index
            )

        track_df["track_id"] = track_df["track_id"] + index_offset
        track_df["tree_id"] = track_df["tree_id"] + index_offset
        if not split_df.empty:
            split_df["parent_track_id"] = split_df["parent_track_id"] + index_offset
            split_df["child_track_id"] = split_df["child_track_id"] + index_offset
        if not merge_df.empty:
            merge_df["parent_track_id"] = merge_df["parent_track_id"] + index_offset
            merge_df["child_track_id"] = merge_df["child_track_id"] + index_offset

        return track_df, split_df, merge_df


def laptrack(coords: Sequence[NumArray], **kwargs) -> nx.Graph:
    """
    Shorthand for calling `LapTrack.fit(coords)`.

    Parameters
    ----------
    coords : Sequence[NumArray]
        The list of coordinates of point for each frame.
        The array index means (sample, dimension).

    **kwargs : dict
        Parameters for the LapTrack initalization

    Returns
    -------
    tracks : networkx.Graph
        The graph for the tracks, whose nodes are (frame, index).

    """
    lt = LapTrack(**kwargs)
    return lt.predict(coords)
