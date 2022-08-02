import networkx as nx
import numpy as np
import pandas as pd

from laptrack import data_conversion


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
            ((0, 0), (1, 1)),
        ]
    )
