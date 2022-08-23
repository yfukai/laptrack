import networkx as nx
import pytest

from laptrack.scores import calc_scores


@pytest.fixture
def test_trees():
    # 0 - 1 - 2 - 3 - 4 - 5
    #           |
    #           - 3 - 4 - 5
    #
    #     1 - 2 - 3 - 4
    true_tree = nx.from_edgelist(
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
        ]
    )

    pred_tree = nx.from_edgelist(
        [
            ((0, 0), (1, 0)),
            ((1, 0), (2, 0)),
            ((2, 0), (3, 0)),
            ((4, 0), (5, 0)),
            ((2, 0), (3, 1)),
            ((1, 2), (2, 2)),
            ((2, 2), (3, 2)),
            ((3, 2), (4, 1)),
            ((4, 1), (5, 1)),
        ]
    )
    return true_tree, pred_tree


def test_scores(test_trees) -> None:
    true_tree, pred_tree = test_trees
    score = {
        "union_ratio": 8 / 12,
        "true_ratio": 8 / 11,
        "predicted_ratio": 8 / 9,
        "track_purity": 7 / 9,
        "target_effectiveness": 6 / 11,
        "division_recovery": 1,
    }
    assert score == calc_scores(true_tree.edges, pred_tree.edges)


def test_scores_no_track(test_trees) -> None:
    true_tree, pred_tree = test_trees
    score = {
        "union_ratio": 8 / 12,
        "true_ratio": 8 / 11,
        "predicted_ratio": 8 / 9,
        "track_purity": -1,
        "target_effectiveness": -1,
        "division_recovery": -1,
    }
    assert score == calc_scores(true_tree.edges, pred_tree.edges, track_scores=False)


def test_scores_exclude(test_trees) -> None:
    true_tree, pred_tree = test_trees
    exclude_edges = [((2, 0), (3, 0)), ((2, 0), (3, 1))]
    score = {
        "union_ratio": 6 / 10,
        "true_ratio": 6 / 9,
        "predicted_ratio": 6 / 7,
        "track_purity": 5 / 7,
        "target_effectiveness": 6 / 9,
        "division_recovery": -1,
    }
    assert score == calc_scores(true_tree.edges, pred_tree.edges, exclude_edges)


def test_scores_exclude2(test_trees) -> None:
    true_tree, pred_tree = test_trees
    exclude_edges = [
        ((2, 0), (3, 0)),
    ]
    score = {
        "union_ratio": 7 / 11,
        "true_ratio": 7 / 10,
        "predicted_ratio": 7 / 8,
        "track_purity": 6 / 8,
        "target_effectiveness": 6 / 10,
        "division_recovery": 1,
    }
    assert score == calc_scores(true_tree.edges, pred_tree.edges, exclude_edges)


def test_scores_include_frames(test_trees) -> None:
    true_tree, pred_tree = test_trees
    exclude_edges = [
        ((2, 0), (3, 0)),
    ]
    include_frames = [0, 2, 3, 4]
    score = {
        "union_ratio": 5 / 9,
        "true_ratio": 5 / 8,
        "predicted_ratio": 5 / 6,
        "track_purity": 4 / 6,
        "target_effectiveness": 4 / 8,
        "division_recovery": 1,
    }
    assert score == calc_scores(
        true_tree.edges, pred_tree.edges, exclude_edges, include_frames=include_frames
    )
