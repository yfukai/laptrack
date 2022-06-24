"""Main module for tracking."""
from enum import Enum
from typing import Callable
from typing import cast
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import TYPE_CHECKING
from typing import Union

# https://stackoverflow.com/questions/59037244/mypy-incompatible-import-error-for-conditional-imports # noqa :
if TYPE_CHECKING:
    from typing import Literal
else:
    try:
        from typing import Literal
    except ImportError:
        from typing_extensions import Literal

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pydantic import BaseModel, Field

from ._cost_matrix import build_frame_cost_matrix, build_segment_cost_matrix
from ._optimization import lap_optimization
from ._typing_utils import FloatArray
from ._typing_utils import Int
from ._utils import coo_matrix_builder


def _get_segment_df(coords, track_tree):
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


class SplittingMergingMode(str, Enum):
    ONE_STEP = "ONE_STEP"
    TWO_STEP = "TWO_STEP"


class LapTrack(BaseModel):
    track_dist_metric: Union[str, Callable] = Field(
        "sqeuclidean",
        description="The metric for calculating track linking cost, "
        + "See documentation for `scipy.spatial.distance.cdist` for accepted values.",
    )
    splitting_dist_metric: Union[str, Callable] = Field(
        "sqeuclidean",
        description="The metric for calculating splitting cost."
        + "See `track_dist_metric`",
    )
    merging_dist_metric: Union[str, Callable] = Field(
        "sqeuclidean",
        description="The metric for calculating merging cost."
        + "See `track_dist_metric`",
    )
    alternative_cost_factor: float = Field(
        1.05,
        description="The factor to calculate the alternative costs"
        + "(b,d,b',d' in Jaqaman et al 2008 NMeth)",
    )
    alternative_cost_percentile: float = Field(
        90,
        description="The percentile to calculate the alternative costs"
        + "(b,d,b',d' in Jaqaman et al 2008 NMeth)",
    )
    alternative_cost_percentile_interpolation: str = Field(
        "lower",
        description="The percentile interpolation to calculate the alternative costs"
        + "(b,d,b',d' in Jaqaman et al 2008 NMeth)."
        + "See `numpy.percentile` for accepted values.",
    )
    track_cost_cutoff: float = Field(
        15**2,
        description="The cost cutoff for the connected points in the track."
        + "For default cases with `dist_metric='sqeuclidean'`,"
        + "this value should be squared maximum distance.",
    )
    track_start_cost: Optional[float] = Field(
        None,  # b in Jaqaman et al 2008 NMeth.
        description="The cost for starting the track (b in Jaqaman et al 2008 NMeth),"
        + "if None, automatically estimated",
    )

    track_end_cost: Optional[float] = Field(
        None,  # b in Jaqaman et al 2008 NMeth.
        description="The cost for ending the track (b in Jaqaman et al 2008 NMeth),"
        + "if None, automatically estimated",
    )

    gap_closing_cost_cutoff: Union[float, Literal[False]] = Field(
        15**2,
        description="The cost cutoff for gap closing."
        + "For default cases with `dist_metric='sqeuclidean'`,"
        + "this value should be squared maximum distance."
        + "If False, no gap closing is allowed.",
    )

    gap_closing_max_frame_count: int = Field(
        2, description="The maximum frame gaps, by default 2."
    )

    splitting_cost_cutoff: Union[int, Literal[False]] = Field(
        False,
        description="The cost cutoff for splitting."
        + "See `gap_closing_cost_cutoff`."
        + "If False, no splitting is allowed.",
    )

    no_splitting_cost: Optional[int] = Field(
        None,  # d' in Jaqaman et al 2008 NMeth.
        description="The cost to reject splitting, if None, automatically estimated.",
    )

    merging_cost_cutoff: Union[int, Literal[False]] = Field(
        False,
        description="The cost cutoff for merging."
        + "See `gap_closing_cost_cutoff`."
        + "If False, no merging is allowed.",
    )

    no_merging_cost: Optional[int] = Field(
        None,  # d' in Jaqaman et al 2008 NMeth.
        description="The cost to reject merging, if None, automatically estimated.",
    )

    splitting_merging_mode: SplittingMergingMode = SplittingMergingMode.ONE_STEP

    def _link_frames(self, coords) -> nx.Graph:
        # initialize tree
        track_tree = nx.Graph()
        for frame, coord in enumerate(coords):
            for j in range(coord.shape[0]):
                track_tree.add_node((frame, j))

        # linking between frames
        for frame, (coord1, coord2) in enumerate(zip(coords[:-1], coords[1:])):
            dist_matrix = cdist(coord1, coord2, metric=self.track_dist_metric)
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
            _, xs, _ = lap_optimization(cost_matrix)

            count1 = dist_matrix.shape[0]
            count2 = dist_matrix.shape[1]
            connections = [(i, xs[i]) for i in range(count1) if xs[i] < count2]
            # track_start=[i for i in range(count1) if xs[i]>count2]
            # track_end=[i for i in range(count2) if ys[i]>count1]
            for connection in connections:
                track_tree.add_edge((frame, connection[0]), (frame + 1, connection[1]))
        return track_tree

    def _get_gap_closing_candidates(self, segments_df):
        if self.gap_closing_cost_cutoff:

            def to_gap_closing_candidates(row):
                target_coord = row["last_frame_coords"]
                frame_diff = segments_df["first_frame"] - row["last_frame"]
                indices = (1 <= frame_diff) & (
                    frame_diff <= self.gap_closing_max_frame_count
                )
                df = segments_df[indices]
                # note: can use KDTree if metric is distance,
                # but might not be appropriate for general metrics
                # https://stackoverflow.com/questions/35459306/find-points-within-cutoff-distance-of-other-points-with-scipy # noqa
                # TrackMate also uses this (trivial) implementation.
                if len(df) > 0:
                    target_dist_matrix = cdist(
                        [target_coord],
                        np.stack(df["first_frame_coords"].values),
                        metric=self.track_dist_metric,
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

            segments_df["gap_closing_candidates"] = segments_df.apply(
                to_gap_closing_candidates, axis=1
            )
        else:
            segments_df["gap_closing_candidates"] = [[]] * len(segments_df)
        return segments_df

    def _get_splitting_merging_candidates(
        self, segments_df, coords, cutoff, prefix, dist_metric
    ):
        if cutoff:

            def to_candidates(row):
                target_coord = row[f"{prefix}_frame_coords"]
                frame = row[f"{prefix}_frame"] + (-1 if prefix == "first" else 1)
                # note: can use KDTree if metric is distance,
                # but might not be appropriate for general metrics
                # https://stackoverflow.com/questions/35459306/find-points-within-cutoff-distance-of-other-points-with-scipy # noqa
                if frame < 0 or len(coords) <= frame:
                    return [], []
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
        return segments_df

    def predict(self, coords) -> nx.Graph:
        if any(list(map(lambda coord: coord.ndim != 2, coords))):
            raise ValueError("the elements in coords must be 2-dim.")
        coord_dim = coords[0].shape[1]
        if any(list(map(lambda coord: coord.shape[1] != coord_dim, coords))):
            raise ValueError("the second dimension in coords must have the same size")

        track_tree = self._link_frames(coords)

        if (
            self.gap_closing_cost_cutoff
            or self.splitting_cost_cutoff
            or self.merging_cost_cutoff
        ):
            segments_df = _get_segment_df(coords, track_tree)

            # compute candidate for gap closing
            segments_df = self._get_gap_closing_candidates(segments_df)

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

            all_candidates: Dict = {}
            dist_matrices: Dict = {}

            # compute candidate for splitting and merging
            for prefix, cutoff, dist_metric in zip(
                ["first", "last"],
                [self.splitting_cost_cutoff, self.merging_cost_cutoff],
                [self.splitting_dist_metric, self.merging_dist_metric],
            ):
                segments_df = self._get_splitting_merging_candidates(
                    segments_df, coords, cutoff, prefix, dist_metric
                )

                all_candidates[prefix] = np.unique(
                    sum(
                        segments_df[f"{prefix}_candidates"].apply(lambda x: list(x[0])),
                        [],
                    ),
                    axis=0,
                )

                N_middle = len(all_candidates[prefix])
                dist_matrices[prefix] = coo_matrix_builder(
                    (N_segments, N_middle), dtype=np.float32
                )

                all_candidates_dict = {
                    tuple(val): i for i, val in enumerate(all_candidates[prefix])
                }
                for ind, row in segments_df.iterrows():
                    candidate_frame_indices = row[f"{prefix}_candidates"][0]
                    candidate_inds = [
                        all_candidates_dict[tuple(fi)] for fi in candidate_frame_indices
                    ]
                    candidate_costs = row[f"{prefix}_candidates"][1]
                    dist_matrices[prefix][
                        (int(cast(Int, ind)), candidate_inds)
                    ] = candidate_costs

            splitting_dist_matrix = dist_matrices["first"]
            merging_dist_matrix = dist_matrices["last"]
            splitting_all_candidates = all_candidates["first"]
            merging_all_candidates = all_candidates["last"]
            cost_matrix = build_segment_cost_matrix(
                gap_closing_dist_matrix,
                splitting_dist_matrix,
                merging_dist_matrix,
                self.track_start_cost,
                self.track_end_cost,
                self.no_splitting_cost,
                self.no_merging_cost,
                self.alternative_cost_factor,
                self.alternative_cost_percentile,
                self.alternative_cost_percentile_interpolation,
            )

            if not cost_matrix is None:
                _, xs, ys = lap_optimization(cost_matrix)

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
                            last_frame_index, tuple(merging_all_candidates[col_ind - M])
                        )

                    row_ind = ys[ind]
                    if M <= row_ind and row_ind < M + N1:
                        track_tree.add_edge(
                            first_frame_index,
                            tuple(splitting_all_candidates[row_ind - M]),
                        )

        return track_tree


def laptrack(coords: Sequence[FloatArray], **kwargs) -> nx.Graph:
    """Track points by solving linear assignment problem.

    Parameters
    ----------
    coords : Sequence[FloatArray]
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
