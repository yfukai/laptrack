from functools import partial
from itertools import product

import numpy as np
import pandas as pd
import pytest
from skimage.measure import regionprops_table

from laptrack import LapTrack
from laptrack import OverLapTrack
from laptrack.datasets import fetch
from laptrack.metric_utils import LabelOverlapOld


@pytest.mark.parametrize(
    "dataset", ["cell_segmentation", "mouse_epidermis", "HL60_3D_synthesized"]
)
@pytest.mark.parametrize("parallel_backend", ["serial", "ray"])
@pytest.mark.parametrize("splitting_cost_cutoff", [False, 0.9])
@pytest.mark.parametrize("merging_cost_cutoff", [False, 0.9])
@pytest.mark.parametrize(
    "track_overlap_coefs", [(1.0, 0.0, 0.0, 0.0, -1.0), (2.0, -0.5, -0.5, -0.5, -0.5)]
)
@pytest.mark.parametrize(
    "splitting_overlap_coefs",
    [(1.0, 0.0, 0.0, 0.0, -1.0), (2.0, -0.5, -0.5, -0.5, -0.5)],
)
def test_overlap_tracking(
    dataset,
    parallel_backend,
    splitting_cost_cutoff,
    merging_cost_cutoff,
    track_overlap_coefs,
    splitting_overlap_coefs,
) -> None:
    if dataset == "mouse_epidermis":
        labels = fetch(dataset)
    else:
        labels = fetch(dataset)["labels"]

    # Old code to calculate overlap tracking
    lo = LabelOverlapOld(labels)
    overlap_records = []
    for f in range(labels.shape[0] - 1):
        print(f)
        l1s = np.unique(labels[f])
        l1s = l1s[l1s != 0]
        l2s = np.unique(labels[f + 1])
        l2s = l2s[l2s != 0]
        for l1, l2 in product(l1s, l2s):
            overlap, iou, ratio_1, ratio_2 = lo.calc_overlap(f, l1, f + 1, l2)
            overlap_records.append(
                {
                    "frame": f,
                    "label1": l1,
                    "label2": l2,
                    "overlap": overlap,
                    "iou": iou,
                    "ratio_1": ratio_1,
                    "ratio_2": ratio_2,
                }
            )
    overlap_df = pd.DataFrame.from_records(overlap_records)
    overlap_df = overlap_df[overlap_df["overlap"] > 0]
    overlap_df = overlap_df.set_index(["frame", "label1", "label2"]).copy()
    dfs = []
    for frame in range(len(labels)):
        df = pd.DataFrame(
            regionprops_table(labels[frame], properties=["label", "centroid"])
        )
        df["frame"] = frame
        dfs.append(df)
    coordinate_df = pd.concat(dfs)

    def metric(c1, c2, params):
        offset, overlap_coef, iou_coef, ratio_1_coef, ratio_2_coef = params
        (frame1, label1), (frame2, label2) = c1, c2
        if frame1 > frame2:
            tmp = (frame1, label1)
            (frame1, label1) = (frame2, label2)
            (frame2, label2) = tmp
        assert frame1 < frame2
        ind = (frame1, label1, label2)
        if ind in overlap_df.index:
            row = overlap_df.loc[ind]
            distance = (
                offset
                + overlap_coef * row["overlap"]
                + iou_coef * row["iou"]
                + ratio_1_coef * row["ratio_1"]
                + ratio_2_coef * row["ratio_2"]
            )
            return distance
        else:
            return offset

    params = dict(
        track_cost_cutoff=0.9,
        gap_closing_max_frame_count=1,
        splitting_cost_cutoff=splitting_cost_cutoff,
        merging_cost_cutoff=merging_cost_cutoff,
    )

    lt = LapTrack(
        track_dist_metric=partial(metric, params=track_overlap_coefs),
        gap_closing_dist_metric=partial(metric, params=track_overlap_coefs),
        splitting_dist_metric=partial(metric, params=splitting_overlap_coefs),
        merging_dist_metric=partial(metric, params=splitting_overlap_coefs),
        **params
    )
    track_df1, split_df1, merge_df1 = lt.predict_dataframe(
        coordinate_df, coordinate_cols=["frame", "label"], only_coordinate_cols=False
    )

    # New tracking
    olt = OverLapTrack(
        parallel_backend=parallel_backend,
        track_dist_metric_coefs=track_overlap_coefs,
        gap_closing_dist_metric_coefs=track_overlap_coefs,
        splitting_dist_metric_coefs=splitting_overlap_coefs,
        merging_dist_metric_coefs=splitting_overlap_coefs,
        **params
    )

    track_df2, split_df2, merge_df2 = olt.predict_overlap_dataframe(labels)

    assert all(track_df1[["tree_id", "track_id"]] == track_df2[["tree_id", "track_id"]])
    assert all(split_df1 == split_df2)
    assert all(merge_df1 == merge_df2)


def test_overlap_tracking_error() -> None:
    lt = LapTrack()
    olt = OverLapTrack()
    with pytest.raises(AttributeError):
        olt.predict_dataframe(pd.DataFrame(), coordinate_cols=["frame", "label"])
    with pytest.raises(AttributeError):
        olt.predict([np.array([1, 2, 3]), np.array([4, 5, 6])])
