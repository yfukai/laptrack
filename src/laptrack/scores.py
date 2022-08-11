"""Tracking score calculation utilities."""
from typing import Dict

import networkx as nx
import pandas as pd

from ._typing_utils import EdgeType
from .data_conversion import convert_tree_to_dataframe
from .utils import order_edges


def _add_split_edges(track_df, split_df):
    track_df2 = track_df.copy()
    for _, row in split_df.iterrows():
        p = (
            track_df[track_df["track_id"] == row["parent_track_id"]]
            .sort_values("frame", ascending=True)
            .iloc[-1]
        )
        p["track_id"] = row["child_track_id"]
        track_df2 = pd.concat([track_df2, pd.DataFrame(p).T])
    track_df2 = track_df2.sort_values(["frame", "index"]).reset_index(drop=True)
    return track_df2


def _df_to_edges(track_df):
    track_edgess = []
    for _, grp in track_df.groupby("track_id"):
        track_edges = []
        nodes = list(grp.sort_values("frame").iterrows())
        for (_, row1), (_, row2) in zip(nodes[:-1], nodes[1:]):
            track_edges.append(
                (tuple(row1[["frame", "index"]]), tuple(row2[["frame", "index"]]))
            )
        track_edgess.append(track_edges)
    return track_edgess


def _calc_overlap_score(reference_edgess, overlap_edgess):
    correct_count = 0
    for reference_edges in reference_edgess:
        overlaps = [
            len(set(reference_edges) & set(overlap_edges))
            for overlap_edges in overlap_edgess
        ]
        max_overlap = max(overlaps)
        correct_count += max_overlap
    return correct_count / sum(
        [len(reference_edges) for reference_edges in reference_edgess]
    )


def calc_scores(
    true_edges: EdgeType, predicted_edges: EdgeType, exclude_true_edges: EdgeType = []
) -> Dict[str, float]:
    """
    Calculate track prediction scores.

    Parameters
    ----------
    true_edges : EdgeType
        the list of true edges. assumes ((frame1,index1), (frame2,index2)) for each edge

    predicted_edges : EdgeType
        the list of predicted edges. see `true_edges` for format

    exclude_true_edges : EdgeType, default []
        the list of true edges to be excluded from "*_ratio". see `true_edges` for format

    Returns
    -------
    Dict[str,float]
        the scores. keys are:
        "union_ratio": (number of TP edges) / (number of TP edges + number of FP edges + number of FN edges)
        "true_ratio": (number of TP edges) / (number of TP edges + number of FN edges)
        "predicted_ratio": (number of TP edges) / (number of TP edges + number of FP edges)
        "track_purity" : the track purity.
        "target_effectiveness" : the target effectiveness.
        "division_recovery" : the number of divisions that were correctly predicted.
    """
    # return the count o
    if len(list(predicted_edges)) == 0:
        return {
            "union_ratio": 0,
            "true_ratio": 0,
            "predicted_ratio": 0,
            "track_purity": 0,
            "target_effectiveness": 0,
            "division_recovery": 0,
        }
    else:
        gt_tree = nx.from_edgelist(order_edges(true_edges), create_using=nx.DiGraph)
        pred_tree = nx.from_edgelist(
            order_edges(predicted_edges), create_using=nx.DiGraph
        )
        gt_track_df, gt_split_df, _gt_merge_df = convert_tree_to_dataframe(gt_tree)
        pred_track_df, pred_split_df, _pred_merge_df = convert_tree_to_dataframe(
            pred_tree
        )
        gt_track_df = gt_track_df.reset_index()
        pred_track_df = pred_track_df.reset_index()

        gt_track_df = _add_split_edges(gt_track_df, gt_split_df)
        pred_track_df = _add_split_edges(pred_track_df, pred_split_df)
        gt_edgess = _df_to_edges(gt_track_df)
        pred_edgess = _df_to_edges(pred_track_df)

        track_purity = _calc_overlap_score(pred_edgess, gt_edgess)
        target_effectiveness = _calc_overlap_score(gt_edgess, pred_edgess)

        def get_children(m):
            return list(gt_tree.successors(m))

        dividing_nodes = [m for m in gt_tree.nodes() if len(get_children(m)) > 1]
        division_recovery_count = 0
        total_count = 0
        for m in dividing_nodes:
            children = get_children(m)

            def check_in(edges):
                return all([(n, m) in edges or (m, n) in edges for n in children])

            excluded = check_in(exclude_true_edges)
            if check_in(predicted_edges) and not excluded:
                division_recovery_count += 1
            if not excluded:
                total_count += 1
        if total_count > 0:
            division_recovery = division_recovery_count / total_count
        else:
            division_recovery = -1

        te = set(true_edges) - set(exclude_true_edges)
        pe = set(predicted_edges) - set(exclude_true_edges)
        return {
            "union_ratio": len(te & pe) / len(te | pe),
            "true_ratio": len(te & pe) / len(te),
            "predicted_ratio": len(te & pe) / len(pe),
            "track_purity": track_purity,
            "target_effectiveness": target_effectiveness,
            "division_recovery": division_recovery,
        }
