"""Data conversion utilities for tracking."""
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd

from ._typing_utils import Int
from ._typing_utils import NumArray
from .utils import _coord_is_empty

IntTuple = Tuple[Int, Int]


def convert_dataframe_to_coords(
    df: pd.DataFrame,
    coordinate_cols: List[str],
    frame_col: str = "frame",
) -> List[NumArray]:
    """
    Convert a track dataframe to a list of coordinates for input.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    coordinate_cols : List[str]
        The list of columns used for the coordinates.
    frame_col : str, default "frame"
        The column name used for the integer frame index.

    Returns
    -------
    coords : List[np.ndarray]
        The list of the coordinates. Note that the first frame is the minimum frame index.
    """
    grps = list(df.groupby(frame_col, sort=True))
    coords_dict = {frame: grp[list(coordinate_cols)].values for frame, grp in grps}
    min_frame = min(coords_dict.keys())
    max_frame = max(coords_dict.keys())
    coords = [
        coords_dict.get(frame, np.array([]))
        for frame in range(min_frame, max_frame + 1)
    ]
    return coords


def convert_dataframe_to_coords_frame_index(
    df: pd.DataFrame,
    coordinate_cols: List[str],
    frame_col: str = "frame",
) -> Tuple[List[NumArray], List[Tuple[int, int]]]:
    """
    Convert a track dataframe to a list of coordinates for input with (frame,index) list.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    coordinate_cols : List[str]
        The list of columns to use for coordinates.
    frame_col : str, default "frame"
        The column name to use for the frame index.

    Returns
    -------
    coords : List[np.ndarray]
        The list of coordinates.
    frame_index : List[Tuple[int, int]]
        The (frame, index) list for the original iloc of the dataframe.
    """
    assert (
        "iloc__" not in df.columns
    ), 'The column name "iloc__" is reserved and cannot be used.'
    df = df.copy()
    df["iloc__"] = np.arange(len(df), dtype=int)

    coords = convert_dataframe_to_coords(
        df, list(coordinate_cols) + ["iloc__"], frame_col
    )

    inverse_map = dict(
        sum(
            [
                [(int(c[-1]), (frame, index)) for index, c in enumerate(coord)]
                for frame, coord in enumerate(coords)
                if not _coord_is_empty(coord)
            ],
            [],
        )
    )

    ilocs = list(range(len(df)))
    assert set(inverse_map.keys()) == set(ilocs)
    frame_index = [inverse_map[i] for i in ilocs]

    return [
        coord[:, :-1] if not _coord_is_empty(coord) else np.array([])
        for coord in coords
    ], frame_index


def convert_tree_to_dataframe(
    tree: nx.DiGraph,
    coords: Optional[Sequence[NumArray]] = None,
    dataframe: Optional[pd.DataFrame] = None,
    frame_index: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert the track tree to dataframes.

    Parameters
    ----------
    tree : nx.Graph
        The track tree, resulted from the traking.
    coords : Optional[Sequence[NumArray]], default None
        The coordinate values. If None, no coordinate values are appended to the dataframe.
    dataframe : Optional[pd.DataFrame], default None
        The dataframe. If not None, `frame_index` should also exist. Ignored if the parameter `coords` is not None.
    frame_index : Optional[List[Tuple[int, int]]], default None
        the inverse map to map (frame, index) to the original iloc of the dataframe.

    Returns
    -------
    track_df : pd.DataFrame
        The track dataframe, with the following columns:

        - "frame" : The frame index.
        - "index" : The coordinate index.
        - "track_id" : The track id.
        - "tree_id" : The tree id.
        - "coord-0", "coord-1", ... : The coordinate values. Exists if coords is not None.
    split_df : pd.DataFrame
        The splitting dataframe, with the following columns:

        - "parent_track_id" : The track id of the parent.
        - "child_track_id" : The track id of the parent.
    merge_df : pd.DataFrame
        The merging dataframe, with the following columns:

        - "parent_track_id" : The track id of the parent.
        - "child_track_id" : The track id of the parent.
    """
    df_data = []
    node_values = np.array(list(tree.nodes))
    frames = np.unique(node_values[:, 0])
    for frame in frames:
        indices = node_values[node_values[:, 0] == frame, 1]
        df_data.append(
            pd.DataFrame(
                {
                    "frame": [frame] * len(indices),
                    "index": indices,
                }
            )
        )
    track_df = pd.concat(df_data)
    if coords is not None:
        # XXX there may exist faster impl.
        for i in range(coords[0].shape[1]):
            track_df[f"coord-{i}"] = [
                coords[int(row["frame"])][int(row["index"]), i]
                for _, row in track_df.iterrows()
            ]
    elif dataframe is not None:
        dataframe = dataframe.copy()
        assert len(track_df) == len(dataframe)
        df_len = len(track_df)
        if frame_index is None:
            raise ValueError("frame_index must not be None if dataframe is not None")
        frame_index_test = set(
            [
                tuple([int(v) for v in vv])
                for vv in track_df[["frame", "index"]].to_numpy()
            ]
        )
        assert (
            set(list(frame_index)) == frame_index_test
        ), "inverse map (frame,index) is incorrect"

        assert "__frame" not in dataframe.columns
        assert "__index" not in dataframe.columns
        dataframe["__frame"] = [x[0] for x in frame_index]
        dataframe["__index"] = [x[1] for x in frame_index]
        track_df = pd.merge(
            track_df,
            dataframe,
            left_on=["frame", "index"],
            right_on=["__frame", "__index"],
            how="outer",
        )
        assert len(track_df) == df_len
        track_df = track_df.drop(columns=["__frame", "__index"]).rename(
            columns={"frame_x": "frame", "index_x": "index"}
        )

    track_df = track_df.set_index(["frame", "index"])
    connected_components = list(nx.connected_components(nx.Graph(tree)))
    for track_id, nodes in enumerate(connected_components):
        for (frame, index) in nodes:
            track_df.loc[(frame, index), "tree_id"] = track_id
    #            tree.nodes[(frame, index)]["tree_id"] = track_id
    tree2 = tree.copy()

    splits: List[Tuple[IntTuple, List[IntTuple]]] = []
    merges: List[Tuple[IntTuple, List[IntTuple]]] = []
    for node in tree.nodes:
        children = list(tree.successors(node))
        parents = list(tree.predecessors(node))
        if len(children) > 1:
            for child in children:
                if tree2.has_edge(node, child):
                    tree2.remove_edge(node, child)
            if node not in [p[0] for p in splits]:
                splits.append((node, children))
        if len(parents) > 1:
            for parent in parents:
                if tree2.has_edge(parent, node):
                    tree2.remove_edge(parent, node)
            if node not in [p[0] for p in merges]:
                merges.append((node, parents))

    connected_components = list(nx.connected_components(nx.Graph(tree2)))
    for track_id, nodes in enumerate(connected_components):
        for (frame, index) in nodes:
            track_df.loc[(frame, index), "track_id"] = track_id
    #            tree.nodes[(frame, index)]["track_id"] = track_id

    for k in ["tree_id", "track_id"]:
        track_df[k] = track_df[k].astype(int)

    split_df_data = []
    for (node, children) in splits:
        for child in children:
            split_df_data.append(
                {
                    "parent_track_id": track_df.loc[node, "track_id"],
                    "child_track_id": track_df.loc[child, "track_id"],
                }
            )
    split_df = pd.DataFrame.from_records(split_df_data).astype(int)

    merge_df_data = []
    for (node, parents) in merges:
        for parent in parents:
            merge_df_data.append(
                {
                    "parent_track_id": track_df.loc[parent, "track_id"],
                    "child_track_id": track_df.loc[node, "track_id"],
                }
            )
    merge_df = pd.DataFrame.from_records(merge_df_data).astype(int)

    return track_df, split_df, merge_df


def convert_split_merge_df_to_napari_graph(
    split_df: pd.DataFrame, merge_df: pd.DataFrame
) -> Dict[int, List[int]]:
    """Convert the split and merge dataframes to a dictionary of parent to children for napari visualization.

    Parameters
    ----------
    split_df : pd.DataFrame
        The splitting dataframe.
    merge_df : pd.DataFrame
        The merging dataframe.

    Returns
    -------
    split_merge_graph : Dict[int,List[int]]
        Dictionary defines the mapping between a track ID and the parents of the track.

    """
    split_merge_graph = {}
    if not split_df.empty:
        split_merge_graph.update(
            {
                int(row["child_track_id"]): [int(row["parent_track_id"])]
                for _, row in split_df.iterrows()
            }
        )
    if not merge_df.empty:
        split_merge_graph.update(
            {
                int(c_id): [int(p_id) for p_id in grp["parent_track_id"]]
                for c_id, grp in merge_df.groupby("child_track_id")
            }
        )
    return split_merge_graph
