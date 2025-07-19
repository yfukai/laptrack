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
    coordinate_cols: Sequence[str],
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


def convert_dataframes_to_tree_coords(
    track_df: pd.DataFrame,
    split_df: pd.DataFrame,
    merge_df: pd.DataFrame,
    coordinate_cols: Sequence[str],
    frame_col: str = "frame",
) -> Tuple[nx.DiGraph, List[NumArray]]:
    """
    Convert the track dataframes to a tree and coordinates.

    Parameters
    ----------
    track_df : pd.DataFrame
        The track dataframe.
    split_df : pd.DataFrame
        The splitting dataframe.
    merge_df : pd.DataFrame
        The merging dataframe.
    coordinate_cols : List[str]
        The list of columns used for the coordinates.
    frame_col : str, default "frame"
        The column name used for the integer frame index.

    Returns
    -------
    tree : nx.DiGraph
        The directed graph representing the track tree.
    coords : List[np.ndarray]
        The list of coordinates.
    """
    _track_df = track_df.sort_values(frame_col).reset_index()
    coords, frame_index = convert_dataframe_to_coords_frame_index(
        _track_df, coordinate_cols, frame_col=frame_col
    )
    frame_index = [(int(frame), int(ind)) for frame, ind in frame_index]
    grp_inds = {
        track_id: grp.index.astype(int)
        for track_id, grp in _track_df.groupby("track_id")
    }

    tree = nx.DiGraph()
    tree.add_nodes_from(frame_index)
    for grp_ind in grp_inds.values():
        if len(grp_ind) > 1:
            for i in range(len(grp_ind) - 1):
                tree.add_edge(frame_index[grp_ind[i]], frame_index[grp_ind[i + 1]])
    for _, row in split_df.iterrows():
        parent_node = grp_inds[row["parent_track_id"]][-1]
        child_node = grp_inds[row["child_track_id"]][0]
        tree.add_edge(
            frame_index[parent_node],
            frame_index[child_node],
        )
    for _, row in merge_df.iterrows():
        parent_node = grp_inds[row["parent_track_id"]][-1]
        child_node = grp_inds[row["child_track_id"]][0]
        tree.add_edge(
            frame_index[parent_node],
            frame_index[child_node],
        )
    return tree, coords


def convert_tree_to_dataframe(
    tree: nx.DiGraph,
    coords: Optional[Sequence[NumArray]] = None,
    dataframe: Optional[pd.DataFrame] = None,
    frame_index: Optional[Sequence[Tuple[int, int]]] = None,
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


def convert_digraph_to_geff_networkx(
    tree: nx.DiGraph,
    coords: Optional[Sequence[NumArray]] = None,
    attr_names: Optional[List[str]] = None,
) -> nx.DiGraph:
    """Convert the networkx directed graph to a networkx in the GEFF format.

    Parameters
    ----------
    tree : nx.DiGraph
        The directed graph representing the track tree.
    coords : Optional[Sequence[NumArray]], default None
        The coordinate values. If None, no coordinate values are appended
        to the dataframe.
    attr_names : Optional[List[str]], default None
        The list of attribute names to be added to the nodes.
        The length must match the number of coordinates in the `coords` + 1.
        If None, default names of "frame", "coord-0", "coord-1", ... are used.

    Returns
    -------
    geff_tree : nx.Graph
        The undirected graph in the GEFF format, with the following attributes.
        - attr_names[0]
        - attr_names[1]
        - ...

    Example
    -------
    >>> import laptrack as lt
    >>> import laptrack.data_conversion as data_conversion
    >>> tree = lt.predict(coords)
    >>> geff_tree = data_conversion.convert_digraph_to_geff_networkx(tree, coords, attr_names)
    >>> geff.write_nx(geff_tree, "save_path.zarr")
    """
    geff_tree = tree.copy()
    for node in geff_tree.nodes:
        geff_tree.nodes[node]["frame"] = node[0]

    # XXX could be more efficient
    if coords is not None:
        if attr_names is None:
            attr_names = ["frame"] + [f"coord-{i}" for i in range(coords[0].shape[1])]
        elif len(attr_names) != coords[0].shape[1] + 1:
            raise ValueError(
                f"attr_names must have length {coords[0].shape[1] + 1}, "
                f"but got {len(attr_names)}"
            )
        for node in geff_tree.nodes:
            for i, attr_name in enumerate(attr_names[1:], start=0):
                geff_tree.nodes[node][attr_name] = coords[node[0]][node[1], i]
    nx.relabel_nodes(
        geff_tree, {node: i for i, node in enumerate(geff_tree.nodes)}, copy=False
    )
    return geff_tree


def convert_dataframes_to_geff_networkx(
    track_df: pd.DataFrame,
    split_df: pd.DataFrame,
    merge_df: pd.DataFrame,
    coordinate_cols: Sequence[str],
    frame_col: str = "frame",
) -> nx.DiGraph:
    """Convert the track dataframes to a GEFF networkx graph.

    Parameters
    ----------
    track_df : pd.DataFrame
        The track dataframe.
    split_df : pd.DataFrame
        The splitting dataframe.
    merge_df : pd.DataFrame
        The merging dataframe.
    coordinate_cols : Sequence[str]
        The list of columns used for the coordinates.
    frame_col : str, default "frame"
        The column name used for the integer frame index.

    Returns
    -------
    geff_tree : nx.DiGraph
        The directed graph in the GEFF format.

    Example
    -------
    >>> import laptrack as lt
    >>> import laptrack.data_conversion as data_conversion
    >>> lt = LapTrack()
    >>> track_df, split_df, merge_df = lt.predict(df, coordinate_cols = ["x","y","z"])
    >>> geff_tree = data_conversion.convert_dataframes_to_geff_networkx(track_df, split_df, merge_df)
    >>> geff.write_nx(geff_tree, "save_path.zarr")
    """
    tree, coords = convert_dataframes_to_tree_coords(
        track_df, split_df, merge_df, coordinate_cols, frame_col
    )
    geff_tree = convert_digraph_to_geff_networkx(
        tree, coords, attr_names=[frame_col] + list(coordinate_cols)
    )
    return geff_tree


def convert_geff_networkx_to_tree_coords(
    geff_tree: nx.DiGraph,
    frame_attr: str = "frame",
    coordinate_attrs: Optional[Sequence[str]] = None,
) -> Tuple[nx.DiGraph, List[NumArray]]:
    """Convert a GEFF networkx graph to a directed graph and coordinates.

    Parameters
    ----------
    geff_tree : nx.DiGraph
        The graph in the GEFF format whose nodes have attributes for frame
        and coordinates.
    frame_attr : str, default "frame"
        The node attribute name that stores the frame index.
    coordinate_attrs : Optional[Sequence[str]], default None
        The node attribute names that store the coordinates. If ``None``, they
        are inferred from the first node excluding ``frame_attr``.

    Returns
    -------
    tree : nx.DiGraph
        Directed graph whose nodes are ``(frame, index)`` tuples.
    coords : List[np.ndarray]
        Coordinate arrays corresponding to each frame.
    """
    # infer coordinate attribute names if not supplied
    if geff_tree.number_of_nodes() == 0:
        return nx.DiGraph(), []

    sample_node, data = next(iter(geff_tree.nodes(data=True)))
    if coordinate_attrs is None:
        coordinate_attrs = [k for k in data.keys() if k != frame_attr]
        coordinate_attrs.sort()

    dim = len(coordinate_attrs)

    # collect nodes for each frame
    frame_to_nodes: Dict[int, List[int]] = {}
    for node, attrs in geff_tree.nodes(data=True):
        frame = int(attrs[frame_attr])
        frame_to_nodes.setdefault(frame, []).append(node)

    frames = sorted(frame_to_nodes.keys())

    coords: List[NumArray] = []
    mapping: Dict[int, Tuple[int, int]] = {}

    for frame in frames:
        nodes = sorted(frame_to_nodes.get(frame, []))
        if nodes:
            coord_arr = np.array(
                [[geff_tree.nodes[n][attr] for attr in coordinate_attrs] for n in nodes]
            )
        else:
            coord_arr = np.array([])
        coords.append(coord_arr)
        for idx, node in enumerate(nodes):
            mapping[node] = (frame, idx)

    tree = nx.DiGraph()
    tree.add_nodes_from(mapping.values())
    for u, v in geff_tree.edges:
        if u in mapping and v in mapping:
            tree.add_edge(mapping[u], mapping[v])

    return tree, coords
