import networkx as nx

from laptrack.scores import calc_scores


def test_scores() -> None:
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

    pred_tree = nx.Graph()
    pred_tree.add_edges_from(
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

    score = {
        "union_ratio": 8 / 12,
        "true_ratio": 8 / 11,
        "predicted_ratio": 8 / 9,
        "track_purity": 7 / 9,
        "target_effectiveness": 6 / 11,
        "division_recovery": 1,
    }

    assert score == calc_scores(true_tree.edges, pred_tree.edges)
