"""Tracking score calculation utilities."""
from typing import Dict
from typing import Optional
from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd

from ._typing_utils import EdgeType
from ._typing_utils import Int
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
    total_count = sum([len(reference_edges) for reference_edges in reference_edgess])
    return correct_count / total_count if total_count > 0 else -1


def calc_scores(
    true_edges: EdgeType,
    predicted_edges: EdgeType,
    exclude_true_edges: EdgeType = [],
    include_frames: Optional[Sequence[Int]] = None,
    track_scores: bool = True,
) -> Dict[str, float]:
    r"""
    Calculate track prediction scores.

    Parameters
    ----------
    true_edges : EdgeType
        The list of true edges. Assumes ((frame1,index1), (frame2,index2)) for each edge.

    predicted_edges : EdgeType
        The list of predicted edges. See `true_edges` for the format.

    exclude_true_edges : EdgeType, default []
        The list of true edges to be excluded from "\*_ratio". See `true_edges` for the format.

    include_frames : Optional[List[Int]], default None
        The list of frames to include in the score calculation. If None, all the frames are included.

    track_scores : bool, default True
        If True, the function calculates track_purity, target_effectiveness and mitotic_branching_correctness.

    Returns
    -------
    score_dict : Dict[str,float]
        The scores in the dict form. The keys are:

        - "Jaccard_index": (number of TP edges) / (number of TP edges + number of FP edges + number of FN edges)
        - "true_positive_rate": (number of TP edges) / (number of TP edges + number of FN edges)
        - "precision": (number of TP edges) / (number of TP edges + number of FP edges)
        - "track_purity" : the track purity.
        - "target_effectiveness" : the target effectiveness.
        - "mitotic_branching_correctness" : the number of divisions that were correctly predicted.
    """
    # return the count o

    if include_frames is None:
        include_frames = list(range(np.max([e[0][0] for e in true_edges]) + 1))
    true_edges_included = [e for e in true_edges if e[0][0] in include_frames]
    predicted_edges_included = [e for e in predicted_edges if e[0][0] in include_frames]

    if len(list(predicted_edges)) == 0:
        return {
            "Jaccard_index": 0,
            "true_positive_rate": 0,
            "precision": 0,
            "track_purity": 0,
            "target_effectiveness": 0,
            "mitotic_branching_correctness": 0,
        }
    else:

        if track_scores:
            ################ calculate track scores #################
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

            filter_edges = (
                lambda e: e[0][0] in include_frames and e not in exclude_true_edges
            )
            pred_edgess = [
                [e for e in edges if filter_edges(e)] for edges in pred_edgess
            ]
            gt_edgess = [[e for e in edges if filter_edges(e)] for edges in gt_edgess]
            track_purity = _calc_overlap_score(pred_edgess, gt_edgess)
            target_effectiveness = _calc_overlap_score(gt_edgess, pred_edgess)

            ################ calculate division recovery #################
            def get_children(m):
                return list(gt_tree.successors(m))

            dividing_nodes = [m for m in gt_tree.nodes() if len(get_children(m)) > 1]
            dividing_nodes = [m for m in dividing_nodes if m[0] in include_frames]
            mitotic_branching_correctness_count = 0
            total_count = 0
            for m in dividing_nodes:
                children = get_children(m)

                def check_match_children(edges):
                    return all([(n, m) in edges or (m, n) in edges for n in children])

                excluded = check_match_children(exclude_true_edges)
                if not excluded:
                    if check_match_children(predicted_edges):
                        mitotic_branching_correctness_count += 1
                    total_count += 1

            if total_count > 0:
                mitotic_branching_correctness = (
                    mitotic_branching_correctness_count / total_count
                )
            else:
                mitotic_branching_correctness = -1
        else:
            track_purity = -1
            target_effectiveness = -1
            mitotic_branching_correctness = -1

        ################ calculate edge overlaps #################
        te = set(true_edges_included) - set(exclude_true_edges)
        pe = set(predicted_edges_included) - set(exclude_true_edges)
        return {
            "Jaccard_index": len(te & pe) / len(te | pe),
            "true_positive_rate": len(te & pe) / len(te),
            "precision": len(te & pe) / len(pe),
            "track_purity": track_purity,
            "target_effectiveness": target_effectiveness,
            "mitotic_branching_correctness": mitotic_branching_correctness,
        }
