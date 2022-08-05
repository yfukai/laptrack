from typing import Dict
from typing import Sequence
from typing import Tuple
from typing import Union

import networkx as nx
import pandas as pd

from ._typing_utils import Int
from .data_conversion import convert_tree_to_dataframe

EdgeType = Union[nx.classes.reportviews.EdgeView, Sequence[Tuple[Int, Int]]]


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


def calc_scores(true_edges: EdgeType, predicted_edges: EdgeType) -> Dict[str, float]:
    """Calculate track prediction scores

    Parameters
    ----------
    true_edges : Sequence[Tuple[Int,Int]]
        the list of true edges. assumes ((frame1,index1), (frame2,index2)) for each edge

    predicted_edges : Sequence[Tuple[Int,Int]]
        the list of predicted edges. assumes ((frame1,index1), (frame2,index2)) for each edge

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
    te = set(true_edges)
    pe = set(predicted_edges)
    if len(pe) == 0:
        return {
            "union_ratio": 0,
            "true_ratio": 0,
            "predicted_ratio": 0,
            "track_purity": 0,
            "target_efficiency": 0,
            "division_recovery": 0,
        }
    else:
        gt_tree = nx.Graph()
        gt_tree.add_edges_from(te)
        pred_tree = nx.Graph()
        pred_tree.add_edges_from(pe)
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
            return [n for n in gt_tree.neighbors(m) if n[0] > m[0]]

        dividing_nodes = [m for m in gt_tree.nodes() if len(get_children(m)) > 1]
        division_recovery_count = 0
        for m in dividing_nodes:
            children = get_children(m)
            if all([(n, m) in pe or (m, n) in pe for n in children]):
                division_recovery_count += 1
        division_recovery = division_recovery_count / len(dividing_nodes)

        return {
            "union_ratio": len(te & pe) / len(te | pe),
            "true_ratio": len(te & pe) / len(te),
            "predicted_ratio": len(te & pe) / len(pe),
            "track_purity": track_purity,
            "target_effectiveness": target_effectiveness,
            "division_recovery": division_recovery,
        }
