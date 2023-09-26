# from functools import cache
from functools import partial
from typing import List
from typing import Tuple
from typing import Union

from pydantic import Field

from ._tracking import LapTrack
from ._typing_utils import IntArray
from .metric_utils import LabelOverlap

CoefType = Tuple[
    float, float, float, float, float
]  # offset, overlap, iou, ratio_1, ratio_2


class OverLapTrack(LapTrack):
    """Tracking by label overlaps."""

    track_dist_metric_coefs: CoefType = Field(
        (1.0, 0.0, 0.0, 0.0, -1.0),
        description="The coefficients to calculate the distance for the overlapping labels."
        + "Must be tuple of 5 floats of `(offset, overlap_coef, iou_coef, ratio_1_coef, ratio_2_coef)`."
        + "The distance is calculated by"
        + "`offset + overlap_coef * overlap + iou_coef * iou + ratio_1_coef * ratio_1 + ratio_2_coef * ratio_2`.",
    )
    gap_closing_dist_metric_coefs: CoefType = Field(
        (1.0, 0.0, 0.0, 0.0, -1.0),
        description="The coefficients to calculate the distance for the overlapping labels."
        + "See `track_dist_metric_coefs` for details.",
    )
    splitting_dist_metric_coefs: CoefType = Field(
        (1.0, 0.0, 0.0, 0.0, -1.0),
        description="The coefficients to calculate the distance for the overlapping labels."
        + "See `track_dist_metric_coefs` for details.",
    )
    merging_dist_metric_coefs: CoefType = Field(
        (1.0, 0.0, 0.0, 0.0, -1.0),
        description="The coefficients to calculate the distance for the overlapping labels."
        + "See `track_dist_metric_coefs` for details.",
    )

    def predict_overlap_dataframe(self, labels: Union[IntArray, List[IntArray]]):
        """Predicts tracks with label overlaps.

        Parameters
        ----------
        labels : Union[IntArray, List[IntArray]]
            Label images.

        Returns
        -------
        track_df : pd.DataFrame
            The track dataframe, with the following columns:

            - "frame" : The frame index.
            - "index" : The coordinate index.
            - "track_id" : The track id.
            - "tree_id" : The tree id.
            - the other columns : The coordinate values.
        split_df : pd.DataFrame
            The splitting dataframe, with the following columns:

            - "parent_track_id" : The track id of the parent.
            - "child_track_id" : The track id of the child.
        merge_df : pd.DataFrame
            The merging dataframe, with the following columns:

            - "parent_track_id" : The track id of the parent.
            - "child_track_id" : The track id of the child.
        """
        lo = LabelOverlap(labels)

        #        @cache
        # XXX Caching does not work with ray
        def _calc_overlap(frame1, label1, frame2, label2):
            return lo.calc_overlap(frame1, label1, frame2, label2)

        def metric(c1, c2, params):
            (frame1, label1), (frame2, label2) = c1, c2
            offset, overlap_coef, iou_coef, ratio_1_coef, ratio_2_coef = params
            if frame1 > frame2:
                tmp = (frame1, label1)
                (frame1, label1) = (frame2, label2)
                (frame2, label2) = tmp
            assert frame1 < frame2
            overlap, iou, ratio_1, ratio_2 = _calc_overlap(
                frame1, label1, frame2, label2
            )
            distance = (
                offset
                + overlap_coef * overlap
                + iou_coef * iou
                + ratio_1_coef * ratio_1
                + ratio_2_coef * ratio_2
            )
            return distance

        self.track_dist_metric = partial(metric, params=self.track_dist_metric_coefs)
        self.gap_closing_dist_metric = partial(
            metric, params=self.gap_closing_dist_metric_coefs
        )
        self.splitting_dist_metric = partial(
            metric, params=self.splitting_dist_metric_coefs
        )
        self.merging_dist_metric = partial(
            metric, params=self.merging_dist_metric_coefs
        )

        track_df, split_df, merge_df = super().predict_dataframe(
            lo.frame_label_df, ["frame", "label"], only_coordinate_cols=False
        )
        assert all(track_df.index.get_level_values("frame") == track_df["frame_y"])
        track_df = (
            track_df.droplevel("frame")
            .rename(columns={"frame_y": "frame"})
            .set_index(["frame", "label"])
        )
        # _calc_overlap.cache_clear()
        return track_df, split_df, merge_df
