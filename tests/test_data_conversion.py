from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd
import pytest

try:
    import geff
except ImportError:
    geff = None

from laptrack import data_conversion
from laptrack import LapTrack
from laptrack._typing_utils import NumArray


def compare_coords_nodes_edges(
    tree1: nx.DiGraph,
    tree2: nx.DiGraph,
    coords1: Sequence[NumArray],
    coords2: Sequence[NumArray],
) -> bool:
    """Compare two trees and their coordinates. The coordinate may be reordered,
    and the index of the (frame, index) tuples for the nodes may differ.

    Parameters
    ----------
    tree1 : nx.DiGraph
        Tree to compare with tree2
    tree2 : nx.DiGraph
        Tree to compare with tree1
    coords1 : Sequence[NumArray]
        Coordinates of the nodes in tree1
    coords2 : Sequence[NumArray]
        Coordinates of the nodes in tree2

    Returns
    -------
    bool
        True if the trees and coordinates are equal, False otherwise.
    """
    if len(coords1) != len(coords2):
        return False
    tree1_coords = {node: tuple(coords1[node[0]][node[1]]) for node in tree1.nodes}
    tree2_coords = {node: tuple(coords2[node[0]][node[1]]) for node in tree2.nodes}
    tree1_2 = nx.relabel_nodes(tree1, tree1_coords, copy=True)
    tree2_2 = nx.relabel_nodes(tree2, tree2_coords, copy=True)
    # XXX the following does not work, but not sure why
    # return nx.is_isomorphic(tree1_2, tree2_2, node_match=lambda x, y: x == y, edge_match=lambda x, y: x == y)
    return set(tree1_2.edges) == set(tree2_2.edges) and set(tree1_2.nodes) == set(
        tree2_2.nodes
    )


def test_compare_coords_nodes_edges():
    tree1 = nx.DiGraph()
    tree2 = nx.DiGraph()

    # Create a simple tree structure
    tree1.add_edges_from([((0, 0), (1, 0)), ((0, 1), (1, 1))])
    tree2.add_edges_from([((0, 0), (1, 1)), ((0, 1), (1, 0))])

    coords1 = [
        np.array([[0.0, 0.0], [0.1, 0.1]]),
        np.array([[1.0, 1.0], [1.1, 1.1]]),
    ]
    coords2 = [
        np.array([[0.0, 0.0], [0.1, 0.1]]),
        np.array([[1.1, 1.1], [1.0, 1.0]]),
    ]

    assert compare_coords_nodes_edges(tree1, tree2, coords1, coords2)
    assert not compare_coords_nodes_edges(tree1, tree2, coords1, coords1)


def test_dataframe_to_coords():
    df = pd.DataFrame(
        {
            "frame": [0, 0, 0, 1, 1, 2, 2, 2, 2, 2],
            "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "z": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    coords_target = [
        np.array([[0, 0], [1, 1], [2, 2]]),
        np.array([[3, 3], [4, 4]]),
        np.array([[5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]),
    ]
    frame_index_target = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
    ]

    coords = data_conversion.dataframe_to_coords(df, ["x", "y"])
    assert len(coords) == len(df["frame"].unique())
    assert all([np.all(c1 == c2) for c1, c2 in zip(coords, coords_target)])

    coords, frame_index = data_conversion.dataframe_to_coords_frame_index(
        df, ["x", "y"]
    )
    assert len(coords) == len(df["frame"].unique())
    assert all([np.all(c1 == c2) for c1, c2 in zip(coords, coords_target)])
    assert frame_index == frame_index_target


@pytest.fixture
def test_trees():
    tree = nx.from_edgelist(
        [
            ((0, 0), (1, 0)),
            ((1, 0), (2, 0)),
            ((2, 0), (3, 0)),
            ((3, 0), (4, 0)),
            ((4, 0), (5, 0)),
            ((2, 0), (3, 1)),
            ((3, 1), (4, 1)),
            ((4, 1), (5, 1)),
            ((1, 2), (2, 2)),
            ((2, 2), (3, 2)),
            ((3, 2), (4, 2)),
            ((1, 3), (2, 2)),
        ],
        create_using=nx.DiGraph,
    )
    tree.add_node((0, 4))
    segments = [
        [(0, 0), (1, 0), (2, 0)],
        [(3, 0), (4, 0), (5, 0)],
        [(3, 1), (4, 1), (5, 1)],
        [(1, 2)],
        [(2, 2), (3, 2), (4, 2)],
        [(1, 3)],
        [(0, 4)],
    ]
    clones = [segments[:3], segments[3:6], segments[6:]]

    # test track graph shape (int .. coordinate index)
    # 0-0-0-0-0-0
    #      |
    #      -1-1-1
    #   2-2-2-2
    #    |
    #   3-
    #
    # 4

    coords = [
        np.array(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [0.2, 0.2],
                [0.3, 0.3],
                [0.4, 0.4],
            ]
        ),
        np.array(
            [
                [1.0, 1.0],
                [1.1, 1.1],
                [1.2, 1.2],
                [1.3, 1.3],
                [1.4, 1.4],
            ]
        ),
        np.array(
            [
                [2.0, 2.0],
                [2.1, 2.1],
                [2.2, 2.2],
                [2.3, 2.3],
                [2.4, 2.4],
            ]
        ),
        np.array(
            [
                [3.0, 3.0],
                [3.1, 3.1],
                [3.2, 3.2],
                [3.3, 3.3],
                [3.4, 3.4],
            ]
        ),
        np.array(
            [
                [4.0, 4.0],
                [4.1, 4.1],
                [4.2, 4.2],
                [4.3, 4.3],
                [4.4, 4.4],
            ]
        ),
        np.array(
            [
                [5.0, 5.0],
                [5.1, 5.1],
                [5.2, 5.2],
                [5.3, 5.3],
                [5.4, 5.4],
            ]
        ),
    ]

    return tree, segments, clones, coords


def test_tree_to_dataframe(test_trees):
    tree, segments, clones, coords = test_trees
    df, split_df, merge_df = data_conversion.tree_to_dataframe(tree, coords)
    len(set(df["track_id"])) == len(segments)

    segment_ids = []
    for segment in segments:
        len(set(df.loc[segment, "track_id"])) == 1  # unique track id
        segment_ids.append(df.loc[segment, "track_id"].iloc[0])
    for clone in clones:
        clone_all = sum(clone, [])
        len(set(df.loc[clone_all, "tree_id"])) == 1  # unique track id

    for i in range(2):
        assert np.allclose(
            df[f"coord-{i}"],
            df.index.get_level_values("frame")
            + 0.1 * df.index.get_level_values("index"),
        )

    split_df_target = np.array(
        [
            [segment_ids[0], segment_ids[1]],
            [segment_ids[0], segment_ids[2]],
        ]
    )
    assert np.all(
        split_df[["parent_track_id", "child_track_id"]].values == split_df_target
    )

    merge_df_target = np.array(
        [
            [segment_ids[3], segment_ids[4]],
            [segment_ids[5], segment_ids[4]],
        ]
    )
    assert np.all(
        merge_df[["parent_track_id", "child_track_id"]].values == merge_df_target
    )


def test_tree_to_dataframe_frame_index():
    df = pd.DataFrame(
        {
            "frame": [0, 0, 0, 1, 1, 2, 2, 2, 2, 2],
            "x": [0.1, 1.1, 2.1, 0.05, 1.05, 0.1, 1.1, 7, 8, 9],
            "y": [0.1, 1.1, 2.1, 0.05, 1.05, 0.1, 1.1, 7, 8, 9],
            "z": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    coords, frame_index = data_conversion.dataframe_to_coords_frame_index(
        df, ["x", "y"]
    )
    lt = LapTrack(gap_closing_max_frame_count=1)
    tree = lt.predict(coords)
    track_df, split_df, merge_df = data_conversion.tree_to_dataframe(
        tree, dataframe=df, frame_index=frame_index
    )
    assert all(df["frame"] == track_df["frame"])
    assert len(np.unique(track_df.iloc[[0, 3, 5]]["tree_id"])) == 1
    assert len(np.unique(track_df.iloc[[1, 4, 6]]["tree_id"])) == 1
    assert len(np.unique(track_df["tree_id"])) > 1


def test_dataframes_to_tree_coords(test_trees):
    tree, segments, clones, coords = test_trees
    track_df, split_df, merge_df = data_conversion.tree_to_dataframe(tree, coords)
    tree2, coords2 = data_conversion.dataframes_to_tree_coords(
        track_df, split_df, merge_df, ["coord-0", "coord-1"], frame_col="frame"
    )
    assert compare_coords_nodes_edges(tree, tree2, coords, coords2)


@pytest.mark.parametrize("track_class", [LapTrack])
def test_integration(track_class):
    df = pd.DataFrame(
        {
            "frame": [0, 0, 0, 1, 1, 2, 2, 2, 2, 2],
            "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "z": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    coords = data_conversion.dataframe_to_coords(df, ["x", "y"])
    lt = track_class()
    tree = lt.predict(coords)
    data_conversion.tree_to_dataframe(tree)


def test_split_merge_df_to_napari_graph(test_trees):
    tree, segments, clones, coords = test_trees
    track_df, split_df, merge_df = data_conversion.tree_to_dataframe(tree, coords)
    graph = data_conversion.split_merge_df_to_napari_graph(split_df, merge_df)

    # test track graph shape (int .. coordinate index)
    # 0-0-0-0-0-0
    #      |
    #      -1-1-1
    #   2-2-2-2
    #    |
    #   3-
    #
    # 4

    # check splitting
    track_id_2_0 = track_df.loc[(2, 0), "track_id"]
    track_id_3_0 = track_df.loc[(3, 0), "track_id"]
    track_id_3_1 = track_df.loc[(3, 1), "track_id"]

    assert graph[track_id_3_0] == [track_id_2_0]
    assert graph[track_id_3_1] == [track_id_2_0]

    # check merging
    track_id_1_2 = track_df.loc[(1, 2), "track_id"]
    track_id_1_3 = track_df.loc[(1, 3), "track_id"]
    track_id_2_2 = track_df.loc[(2, 2), "track_id"]

    assert graph[track_id_2_2] == [track_id_1_2, track_id_1_3]

    # only split
    graph = data_conversion.split_merge_df_to_napari_graph(split_df, pd.DataFrame())
    assert graph[track_id_3_0] == [track_id_2_0]
    assert graph[track_id_3_1] == [track_id_2_0]

    # only merge
    graph = data_conversion.split_merge_df_to_napari_graph(pd.DataFrame(), merge_df)
    assert graph[track_id_2_2] == [track_id_1_2, track_id_1_3]


@pytest.mark.skipif(geff is None, reason="geff is not installed")
def test_to_geff_networkx(test_trees, tmp_path):
    tree, segments, clones, coords = test_trees
    geff_trees = data_conversion.digraph_to_geff_networkx(
        tree, coords, ["frame", "x", "y"]
    )
    geff.write_nx(geff_trees.tree, tmp_path / "test.geff")
    geff_tree2, metadata = geff.read_nx(tmp_path / "test.geff")
    assert set(geff_trees.tree.nodes) == set(geff_tree2.nodes)
    assert set(geff_trees.tree.edges) == set(geff_tree2.edges)


@pytest.mark.skipif(geff is None, reason="geff is not installed")
def test_geff_networkx_to_tree_coords_with_mapping(test_trees):
    tree, _, _, coords = test_trees
    geff_trees = data_conversion.digraph_to_geff_networkx(
        tree, coords, ["frame", "x", "y"]
    )
    tree2, coords2, mapping = data_conversion.geff_networkx_to_tree_coords_mapping(
        geff_trees.tree, frame_attr="frame", coordinate_attrs=["x", "y"]
    )
    assert compare_coords_nodes_edges(tree, tree2, coords, coords2)
    assert len(mapping) == len(geff_trees.tree.nodes)

    def get_data_from_node(node):
        return coords2[node[0]][node[1]]

    for node in geff_trees.tree.nodes:
        d1 = [geff_trees.tree.nodes[node].get(attr) for attr in ["x", "y"]]
        n2 = mapping[node]
        assert geff_trees.tree.nodes[node]["frame"] == n2[0]
        assert np.all(d1 == get_data_from_node(n2))
