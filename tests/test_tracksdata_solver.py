"""Test cases for the tracksdata solver."""

import numpy as np
import pytest

from laptrack import LapTrack

pytest.importorskip("tracksdata")

import polars as pl  # noqa: E402
from tracksdata.constants import DEFAULT_ATTR_KEYS  # noqa: E402
from tracksdata.graph import RustWorkXGraph  # noqa: E402

from laptrack.tracksdata_solver import LapTrackSolver  # noqa: E402


@pytest.fixture
def coords():
    np.random.seed(0)
    return [np.random.random((5, 2)) * 100 for _ in range(4)]


@pytest.fixture
def graph(coords):
    graph = RustWorkXGraph()
    for key in ["x", "y"]:
        graph.add_node_attr_key(key, pl.Float64)
    for frame, coord in enumerate(coords):
        for x, y in coord:
            graph.add_node({"t": frame, "x": float(x), "y": float(y)})
    return graph


def _expected_edges(graph, coords, lt):
    node_df = graph.node_attrs(
        attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, "t", "x", "y"]
    ).sort(["t", DEFAULT_ATTR_KEYS.NODE_ID])
    node_ids = {}
    for frame in sorted(set(node_df["t"].to_list())):
        sub_df = node_df.filter(pl.col("t") == frame)
        for index, node_id in enumerate(sub_df[DEFAULT_ATTR_KEYS.NODE_ID].to_list()):
            node_ids[(frame, index)] = node_id
    tree = lt.predict(coords)
    return set((node_ids[edge[0]], node_ids[edge[1]]) for edge in tree.edges())


def test_solver_adds_solution(graph, coords):
    lt = LapTrack(cutoff=1000**2)
    solver = LapTrackSolver(tracker=lt, coordinate_attr_keys=["x", "y"])
    solution = solver.solve(graph)

    expected_edges = _expected_edges(graph, coords, lt)

    edges_df = graph.edge_attrs()
    solution_edges = set(
        (int(s), int(t))
        for s, t, sol in zip(
            edges_df[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_numpy(),
            edges_df[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_numpy(),
            edges_df[DEFAULT_ATTR_KEYS.SOLUTION].to_numpy(),
        )
        if sol
    )
    assert solution_edges == expected_edges

    # the returned view only contains the solution
    assert solution is not None
    assert solution.num_edges() == len(expected_edges)


def test_solver_infers_coordinate_keys(graph, coords):
    lt = LapTrack(cutoff=1000**2)
    solver = LapTrackSolver(tracker=lt)
    solver.solve(graph)
    keys = solver._get_coordinate_attr_keys(graph)
    assert keys == ["y", "x"]


def test_solver_marks_existing_edges(graph, coords):
    lt = LapTrack(cutoff=1000**2)
    expected_edges = _expected_edges(graph, coords, lt)

    graph.add_edge_attr_key("distance", pl.Float64)
    source, target = next(iter(expected_edges))
    edge_id = graph.add_edge(source, target, {"distance": 0.0})

    solver = LapTrackSolver(tracker=lt, coordinate_attr_keys=["x", "y"])
    solver.solve(graph)

    edges_df = graph.edge_attrs()
    row = edges_df.filter(pl.col(DEFAULT_ATTR_KEYS.EDGE_ID) == edge_id)
    assert row[DEFAULT_ATTR_KEYS.SOLUTION].to_list() == [True]
    # no duplicated edge is added
    pairs = list(
        zip(
            edges_df[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list(),
            edges_df[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list(),
        )
    )
    assert len(pairs) == len(set(pairs))


def test_solver_empty_graph():
    graph = RustWorkXGraph()
    solver = LapTrackSolver()
    with pytest.raises(ValueError):
        solver.solve(graph)


def test_solver_preserved_edges():
    graph = RustWorkXGraph()
    for key in ["x", "y"]:
        graph.add_node_attr_key(key, pl.Float64)
    n00 = graph.add_node({"t": 0, "x": 0.0, "y": 0.0})
    n01 = graph.add_node({"t": 0, "x": 100.0, "y": 100.0})
    n10 = graph.add_node({"t": 1, "x": 0.5, "y": 0.0})
    n11 = graph.add_node({"t": 1, "x": 200.0, "y": 200.0})
    graph.add_edge_attr_key("preserved", pl.Boolean)
    # n01 -> n11 is far beyond the cutoff, but marked as preserved
    preserved_edge_id = graph.add_edge(n01, n11, {"preserved": True})
    graph.add_edge(n00, n11, {"preserved": False})

    lt = LapTrack(
        cutoff=5.0**2,
        gap_closing_cutoff=False,
        splitting_cutoff=False,
        merging_cutoff=False,
    )
    solver = LapTrackSolver(
        tracker=lt,
        coordinate_attr_keys=["x", "y"],
        connected_edge_attr_key="preserved",
    )
    solver.solve(graph)

    edges_df = graph.edge_attrs()
    solution_edges = set(
        (int(s), int(t))
        for s, t, sol in zip(
            edges_df[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_numpy(),
            edges_df[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_numpy(),
            edges_df[DEFAULT_ATTR_KEYS.SOLUTION].to_numpy(),
        )
        if sol
    )
    # the preserved edge is kept even though it is far beyond the cutoff,
    # and the nearby pair is still linked normally
    assert solution_edges == {(n01, n11), (n00, n10)}
    row = edges_df.filter(pl.col(DEFAULT_ATTR_KEYS.EDGE_ID) == preserved_edge_id)
    assert row[DEFAULT_ATTR_KEYS.SOLUTION].to_list() == [True]


def test_solver_preserved_edges_missing_key(coords, graph):
    solver = LapTrackSolver(connected_edge_attr_key="preserved")
    with pytest.raises(ValueError, match="does not exist in the graph"):
        solver.solve(graph)
