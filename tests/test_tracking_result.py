"""Test cases for the TrackingResult class."""

import networkx as nx
import pandas as pd
import pytest

from laptrack import LapTrack
from laptrack import TrackingResult


@pytest.fixture
def spots_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": [0, 0, 1, 1, 2, 2, 2],
            "x": [10.0, 50.0, 11.0, 51.0, 12.0, 45.0, 60.0],
            "y": [10.0, 50.0, 11.0, 51.0, 12.0, 55.0, 51.0],
        }
    )


@pytest.fixture
def lt() -> LapTrack:
    return LapTrack(
        cutoff=15**2,
        splitting_cutoff=20**2,
        merging_cutoff=False,
    )


def test_to_dataframes_matches_predict_dataframe(spots_df, lt) -> None:
    result = lt.predict_tracking_result(spots_df, ["x", "y"])
    assert isinstance(result, TrackingResult)
    track_df, split_df, merge_df = result.to_dataframes()
    track_df2, split_df2, merge_df2 = lt.predict_dataframe(spots_df, ["x", "y"])
    pd.testing.assert_frame_equal(track_df, track_df2)
    pd.testing.assert_frame_equal(split_df, split_df2)
    pd.testing.assert_frame_equal(merge_df, merge_df2)

    # index offset
    track_df3, split_df3, merge_df3 = result.to_dataframes(index_offset=5)
    assert track_df3["track_id"].min() == 5
    assert track_df3["tree_id"].min() == 5
    if not split_df3.empty:
        assert (split_df3["parent_track_id"] == split_df["parent_track_id"] + 5).all()
        assert (split_df3["child_track_id"] == split_df["child_track_id"] + 5).all()


def test_to_indices_networkx(spots_df, lt) -> None:
    result = lt.predict_tracking_result(spots_df, ["x", "y"])
    tree = result.to_indices_networkx()
    assert isinstance(tree, nx.DiGraph)
    assert set(tree.nodes) == set(result.tree.nodes)
    assert set(tree.edges) == set(result.tree.edges)
    # the returned graph is a copy
    tree.add_node((100, 0))
    assert (100, 0) not in result.tree.nodes


def test_to_geff_networkx(spots_df, lt) -> None:
    result = lt.predict_tracking_result(spots_df, ["x", "y"])
    geff_tree = result.to_geff_networkx()
    assert geff_tree.number_of_nodes() == len(spots_df)
    assert geff_tree.number_of_edges() == result.tree.number_of_edges()
    for _node, data in geff_tree.nodes(data=True):
        assert set(data.keys()) >= {"frame", "x", "y"}
    # node labels are integers
    assert all(isinstance(n, int) for n in geff_tree.nodes)

    # custom attribute names
    geff_tree2 = result.to_geff_networkx(attr_names=["t", "pos_x", "pos_y"])
    for _node, data in geff_tree2.nodes(data=True):
        assert set(data.keys()) >= {"t", "pos_x", "pos_y"}


def test_coords_only_result(spots_df, lt) -> None:
    coords = [grp[["x", "y"]].values for _, grp in spots_df.groupby("frame", sort=True)]
    tree = lt.predict(coords)
    result = TrackingResult(tree=tree, coords=coords)
    track_df, split_df, merge_df = result.to_dataframes()
    assert len(track_df) == len(spots_df)
    assert {"coord-0", "coord-1", "track_id", "tree_id"} <= set(track_df.columns)
    geff_tree = result.to_geff_networkx()
    for _node, data in geff_tree.nodes(data=True):
        assert set(data.keys()) >= {"frame", "coord-0", "coord-1"}


def test_write_csvs(spots_df, lt, tmp_path) -> None:
    result = lt.predict_tracking_result(spots_df, ["x", "y"])
    paths = result.write_csvs(tmp_path / "outputs", prefix="test_")
    assert set(paths.keys()) == {"track", "split", "merge"}
    for name, path in paths.items():
        assert path.exists()
        assert path.name == f"test_{name}.csv"
    track_df, split_df, merge_df = result.to_dataframes()
    track_df_read = pd.read_csv(paths["track"])
    pd.testing.assert_frame_equal(
        track_df_read, track_df.reset_index(drop=True), check_dtype=False
    )
    split_df_read = pd.read_csv(paths["split"])
    assert len(split_df_read) == len(split_df)


def test_write_geff(spots_df, lt, tmp_path) -> None:
    geff = pytest.importorskip("geff")
    result = lt.predict_tracking_result(spots_df, ["x", "y"])
    geff_path = tmp_path / "tracks.geff"
    result.write_geff(geff_path)
    graph, _metadata = geff.read(geff_path)
    assert graph.number_of_nodes() == len(spots_df)
    assert graph.number_of_edges() == result.tree.number_of_edges()


def test_to_dataframes_requires_frame_index(spots_df, lt) -> None:
    result = lt.predict_tracking_result(spots_df, ["x", "y"])
    result_broken = TrackingResult(
        tree=result.tree,
        dataframe=spots_df,
        frame_index=None,
    )
    with pytest.raises(ValueError):
        result_broken.to_dataframes()
