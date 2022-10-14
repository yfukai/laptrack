# %%
"""Test cases for the tracking."""
import warnings
from itertools import product
from os import path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from laptrack import LapTrack
from laptrack import laptrack
from laptrack import LapTrackMulti
from laptrack.data_conversion import convert_tree_to_dataframe

warnings.simplefilter("ignore", FutureWarning)

DEFAULT_PARAMS = dict(
    track_dist_metric="sqeuclidean",
    splitting_dist_metric="sqeuclidean",
    merging_dist_metric="sqeuclidean",
    track_cost_cutoff=15**2,
    track_start_cost=None,
    track_end_cost=None,
    gap_closing_cost_cutoff=15**2,
    gap_closing_max_frame_count=2,
    splitting_cost_cutoff=15**2,
    no_splitting_cost=None,
    merging_cost_cutoff=15**2,
    no_merging_cost=None,
)

FILENAME_SUFFIX_PARAMS = [
    (
        "without_gap_closing",
        {
            **DEFAULT_PARAMS,  # type: ignore
            "gap_closing_cost_cutoff": False,
            "splitting_cost_cutoff": False,
            "merging_cost_cutoff": False,
        },
    ),
    (
        "with_gap_closing",
        {
            **DEFAULT_PARAMS,  # type: ignore
            "splitting_cost_cutoff": False,
            "merging_cost_cutoff": False,
        },
    ),
    (
        "with_splitting",
        {
            **DEFAULT_PARAMS,  # type: ignore
            "merging_cost_cutoff": False,
        },
    ),
    (
        "with_merging",
        {
            **DEFAULT_PARAMS,  # type: ignore
            "splitting_cost_cutoff": False,
        },
    ),
    (
        "with_splitting_merging",
        {
            **DEFAULT_PARAMS,  # type: ignore
            "alternative_cost_percentile_interpolation": "higher",
        },
    ),
]  # type: ignore


@pytest.fixture(params=FILENAME_SUFFIX_PARAMS)
def testdata(request, shared_datadir: str):
    filename_suffix, params = request.param
    filename = path.join(shared_datadir, f"trackmate_tracks_{filename_suffix}")
    spots_df = pd.read_csv(filename + "_spots.csv")
    frame_max = spots_df["frame"].max()
    coords = []
    spot_ids = []
    for i in range(frame_max):
        df = spots_df[spots_df["frame"] == i]
        coords.append(df[["position_x", "position_y"]].values)
        spot_ids.append(df["id"].values)

    spot_id_to_coord_id = {}
    for i, spot_ids_frame in enumerate(spot_ids):
        for j, spot_id in enumerate(spot_ids_frame):
            assert not spot_id in spot_id_to_coord_id
            spot_id_to_coord_id[spot_id] = (i, j)

    edges_df = pd.read_csv(filename + "_edges.csv", index_col=0)
    edges_df["coord_source_id"] = edges_df["spot_source_id"].map(spot_id_to_coord_id)
    edges_df["coord_target_id"] = edges_df["spot_target_id"].map(spot_id_to_coord_id)
    valid_edges_df = edges_df[~pd.isna(edges_df["coord_target_id"])]
    edges_arr = valid_edges_df[["coord_source_id", "coord_target_id"]].values
    edges_set = set(list(map(tuple, (edges_arr))))

    return params, coords, edges_set


@pytest.mark.parametrize("tracker_class", [LapTrack, LapTrackMulti])
def test_reproducing_trackmate(testdata, tracker_class) -> None:
    params, coords, edges_set = testdata
    lt = tracker_class(**params)
    track_tree = lt.predict(coords)
    assert edges_set == set(track_tree.edges)
    for n in track_tree.nodes():
        for m in track_tree.successors(n):
            assert m[0] > n[0]

    data = []
    for frame, c in enumerate(coords):
        assert c.shape[1] == 2
        data.append(
            pd.DataFrame(
                {
                    "frame": [frame] * len(c),
                    "x": c[:, 0],
                    "y": c[:, 1],
                }
            )
        )
    df = pd.concat(data)
    track_df, split_df, merge_df = lt.predict_dataframe(df, ["x", "y"])
    track_df2, split_df2, merge_df2 = convert_tree_to_dataframe(track_tree, coords)
    track_df2 = track_df2.rename(columns={"coord-0": "x", "coord-1": "y"})
    assert all(track_df == track_df2)
    assert all(split_df == split_df2)
    assert all(merge_df == merge_df2)

    track_df, split_df, merge_df = lt.predict_dataframe(
        df, ["x", "y"], only_coordinate_cols=False
    )
    assert all(track_df["frame_y"] == track_df2.index.get_level_values("frame"))
    track_df = track_df.drop(columns=["frame_y"])
    assert all(track_df == track_df2)
    assert all(split_df == split_df2)
    assert all(merge_df == merge_df2)


@pytest.fixture(params=[2, 3, 4])
def dist_metric(request):
    if request.param == 2:
        return lambda c1, c2: np.linalg.norm(c1 - c2) ** 2
    elif request.param == 3:
        return lambda c1, c2, _1: np.linalg.norm(c1 - c2) ** 2
    elif request.param == 4:
        return lambda c1, c2, _1, _2: np.linalg.norm(c1 - c2) ** 2


def test_multi_algorithm_reproducing_trackmate_lambda(testdata, dist_metric) -> None:
    params, coords, edges_set = testdata
    params = params.copy()
    params.update(
        dict(
            track_dist_metric=lambda c1, c2: np.linalg.norm(c1 - c2) ** 2,
            splitting_dist_metric=dist_metric,
            merging_dist_metric=dist_metric,
        )
    )
    lt = LapTrackMulti(**params)
    track_tree = lt.predict(coords)
    assert edges_set == set(track_tree.edges)


def test_multi_algorithm_reproducing_trackmate_3_arg_lambda(testdata) -> None:
    params, coords, edges_set = testdata
    lt = LapTrackMulti(**params)
    track_tree = lt.predict(coords)
    assert edges_set == set(track_tree.edges)


def test_multi_algorithm_reproducing_trackmate_4_arg_lambda(testdata) -> None:
    params, coords, edges_set = testdata
    lt = LapTrackMulti(**params)
    track_tree = lt.predict(coords)
    assert edges_set == set(track_tree.edges)


def test_laptrack_function_shortcut(testdata) -> None:
    params, coords, edges_set = testdata
    lt = LapTrack(**params)
    track_tree1 = lt.predict(coords)
    track_tree2 = laptrack(coords, **params)
    assert set(track_tree1.edges) == set(track_tree2.edges)


def test_tracking_zero_distance() -> None:
    coords = [np.array([[10, 10], [12, 11]]), np.array([[10, 10], [13, 11]])]
    lt = LapTrack(
        gap_closing_cost_cutoff=False,
        splitting_cost_cutoff=False,
        merging_cost_cutoff=False,
    )  # type: ignore
    track_tree = lt.predict(coords)
    edges = track_tree.edges()
    assert set(edges) == set([((0, 0), (1, 0)), ((0, 1), (1, 1))])


def test_tracking_not_connected() -> None:
    coords = [np.array([[10, 10], [12, 11]]), np.array([[50, 50], [53, 51]])]
    lt = LapTrack(
        track_cost_cutoff=15**2,
        gap_closing_cost_cutoff=False,
        splitting_cost_cutoff=False,
        merging_cost_cutoff=False,
    )  # type: ignore
    track_tree = lt.predict(coords)
    edges = track_tree.edges()
    assert set(edges) == set()


def test_gap_closing(shared_datadir: str) -> None:
    coords = list(
        np.load(
            path.join(shared_datadir, "grouped_poss_molecule_tracking.npy"),
            allow_pickle=True,
        )
    )
    lt = LapTrack(
        track_cost_cutoff=15**2,
        splitting_cost_cutoff=False,
        merging_cost_cutoff=False,
    )  # type: ignore
    track_tree = lt.predict(coords)
    for track in nx.connected_components(nx.Graph(track_tree)):
        frames, _ = zip(*track)
        assert len(set(frames)) == len(frames)


def test_no_accepting_wrong_argments() -> None:
    with pytest.raises(ValidationError):
        lt = LapTrack(hogehoge=True)
    with pytest.raises(ValidationError):
        lt = LapTrack(fugafuga=True)


@pytest.mark.parametrize("tracker_class", [LapTrack, LapTrackMulti])
def test_connected_edges(tracker_class) -> None:
    coords = [np.array([[10, 10], [12, 11]]), np.array([[10, 10], [13, 11]])]
    lt = tracker_class(
        gap_closing_cost_cutoff=100,
        splitting_cost_cutoff=100,
        merging_cost_cutoff=100,
    )  # type: ignore
    connected_edges = [((0, 0), (1, 1))]
    track_tree = lt.predict(coords, connected_edges=connected_edges)
    edges = track_tree.edges()
    assert set(edges) == set([((0, 0), (1, 1)), ((0, 1), (1, 0))])


@pytest.mark.parametrize("tracker_class", [LapTrack, LapTrackMulti])
def test_connected_edges_splitting(tracker_class) -> None:
    coords = [
        np.array([[10, 10], [11, 11], [13, 12]]),
        np.array([[10, 10], [13, 11], [13, 15]]),
    ]
    lt = tracker_class(
        gap_closing_cost_cutoff=100,
        splitting_cost_cutoff=100,
        merging_cost_cutoff=False,
    )  # type: ignore
    connected_edges = [((0, 0), (1, 1)), ((0, 0), (1, 2))]
    track_tree = lt.predict(coords, connected_edges=connected_edges)
    edges = track_tree.edges()
    assert set(edges) == set([((0, 0), (1, 1)), ((0, 0), (1, 2)), ((0, 1), (1, 0))])


@pytest.mark.parametrize("tracker_class", [LapTrack, LapTrackMulti])
def test_no_connected_node(tracker_class) -> None:
    coords = [np.array([[10, 10], [12, 11]]), np.array([[10, 10], [100, 11]])]
    lt = tracker_class(
        gap_closing_cost_cutoff=1,
    )  # type: ignore
    track_tree = lt.predict(coords)
    for frame, index in product([0, 1], [0, 1]):
        assert (frame, index) in track_tree.nodes()


# # %%
# filename_suffix, params = FILENAME_SUFFIX_PARAMS[-1]
# # params['splitting_cost_cutoff']=False
# #params['merging_cost_cutoff']=50**2
# filename = path.join("data/", f"trackmate_tracks_{filename_suffix}")
# spots_df = pd.read_csv(filename + "_spots.csv")
# frame_max = spots_df["frame"].max()
# coords = []
# spot_ids = []
# for i in range(frame_max):
#     df = spots_df[spots_df["frame"] == i]
#     coords.append(df[["position_x", "position_y"]].values)
#     spot_ids.append(df["id"].values)
#
# spot_id_to_coord_id = {}
# for i, spot_ids_frame in enumerate(spot_ids):
#     for j, spot_id in enumerate(spot_ids_frame):
#         assert not spot_id in spot_id_to_coord_id
#         spot_id_to_coord_id[spot_id] = (i, j)
#
# edges_df = pd.read_csv(filename + "_edges.csv", index_col=0)
# edges_df["coord_source_id"] = edges_df["spot_source_id"].map(spot_id_to_coord_id)
# edges_df["coord_target_id"] = edges_df["spot_target_id"].map(spot_id_to_coord_id)
# valid_edges_df = edges_df[~pd.isna(edges_df["coord_target_id"])]
# edges_arr = valid_edges_df[["coord_source_id", "coord_target_id"]].values
# edges_set = set(list(map(tuple, (edges_arr))))
#
# lt = LapTrack(**params)
# track_tree = lt.predict(coords)
# assert edges_set == set(track_tree.edges)
# # %%
#
