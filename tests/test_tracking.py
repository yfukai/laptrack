"""Test cases for the tracking."""
from os import path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from laptrack import LapTrack
from laptrack import laptrack
from laptrack import LapTrackMulti

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


def test_reproducing_trackmate(testdata) -> None:
    params, coords, edges_set = testdata
    lt = LapTrack(**params)
    track_tree = lt.predict(coords)
    assert edges_set == set(track_tree.edges)


def test_multi_algorithm_reproducing_trackmate(testdata) -> None:
    params, coords, edges_set = testdata
    lt = LapTrackMulti(**params)
    track_tree = lt.predict(coords)
    assert edges_set == set(track_tree.edges)


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
    for track in nx.connected_components(track_tree):
        frames, _ = zip(*track)
        assert len(set(frames)) == len(frames)


def test_no_accepting_wrong_argments() -> None:
    with pytest.raises(ValidationError):
        lt = LapTrack(hogehoge=True)
    with pytest.raises(ValidationError):
        lt = LapTrack(fugafuga=True)


def test_connected_edges() -> None:
    coords = [np.array([[10, 10], [12, 11]]), np.array([[10, 10], [13, 11]])]
    lt = LapTrack(
        gap_closing_cost_cutoff=100,
        splitting_cost_cutoff=100,
        merging_cost_cutoff=100,
    )  # type: ignore
    connected_edges = [((0, 0), (1, 1))]
    track_tree = lt.predict(coords, connected_edges=connected_edges)
    edges = track_tree.edges()
    assert set(edges) == set([((0, 0), (1, 1)), ((0, 1), (1, 0))])


def test_connected_edges_splitting() -> None:
    coords = [
        np.array([[10, 10], [11, 11], [13, 12]]),
        np.array([[10, 10], [13, 11], [13, 15]]),
    ]
    lt = LapTrack(
        gap_closing_cost_cutoff=100,
        splitting_cost_cutoff=100,
        merging_cost_cutoff=100,
    )  # type: ignore
    connected_edges = [((0, 0), (1, 1)), ((0, 0), (1, 2))]
    track_tree = lt.predict(coords, connected_edges=connected_edges)
    edges = track_tree.edges()
    assert set(edges) == set([((0, 0), (1, 1)), ((0, 0), (1, 2)), ((0, 1), (1, 0))])
