import networkx as nx
import numpy as np
import pandas as pd
import pytest

from laptrack import data_conversion
from laptrack import LapTrack
from laptrack import LapTrackMulti


def test_convert_dataframe_to_coords():
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

    coords = data_conversion.convert_dataframe_to_coords(df, ["x", "y"])
    assert len(coords) == len(df["frame"].unique())
    assert all([np.all(c1 == c2) for c1, c2 in zip(coords, coords_target)])


def test_convert_tree_to_dataframe():
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
    segments = [
        [(0, 0), (1, 0), (2, 0)],
        [(3, 0), (4, 0), (5, 0)],
        [(3, 1), (4, 1), (5, 1)],
        [(1, 2)],
        [(2, 2), (3, 2), (4, 2)],
        [(1, 3)],
    ]
    clones = [segments[:3], segments[3:]]
    # 0-0-0-0-0-0
    #      |
    #      -1-1-1
    #   2-2-2-2
    #    |
    #   3-
    df, split_df, merge_df = data_conversion.convert_tree_to_dataframe(tree)
    len(set(df["track_id"])) == len(segments)
    segment_ids = []
    for segment in segments:
        len(set(df.loc[segment, "track_id"])) == 1  # unique track id
        segment_ids.append(df.loc[segment, "track_id"].iloc[0])
    for clone in clones:
        clone_all = sum(clone, [])
        len(set(df.loc[clone_all, "tree_id"])) == 1  # unique track id

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


@pytest.mark.parametrize("track_class", [LapTrack, LapTrackMulti])
def test_integration(track_class):
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

    coords = data_conversion.convert_dataframe_to_coords(df, ["x", "y"])
    lt = track_class()
    tree = lt.predict(coords)
    data_conversion.convert_tree_to_dataframe(tree)
