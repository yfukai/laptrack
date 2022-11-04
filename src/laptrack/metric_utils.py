"""Utilities for metric calculation."""
from typing import Tuple

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from ._typing_utils import Float
from ._typing_utils import Int
from ._typing_utils import IntArray


class LabelOverlap:
    """Utility object to calculate overlap of segmentation labels between frames."""

    def _intersect_bbox(self, r1, r2):
        bbox = []
        for i in range(self.ndim):
            y0 = max(r1[f"bbox-{i}"], r2[f"bbox-{i}"])
            y1 = min(r1[f"bbox-{i+self.ndim}"], r2[f"bbox-{i+self.ndim}"])
            bbox.append((y0, y1))
        if all([y0 <= y1 for y0, y1 in bbox]):
            return bbox
        else:
            return None

    def _union_bbox(self, r1, r2):
        bbox = []
        for i in range(self.ndim):
            y0 = min(r1[f"bbox-{i}"], r2[f"bbox-{i}"])
            y1 = max(r1[f"bbox-{i+self.ndim}"], r2[f"bbox-{i+self.ndim}"])
            bbox.append((y0, y1))
        return bbox

    def __init__(self, label_images: IntArray):
        """Summarise the segmentation properties and initialize the object.

        Parameters
        ----------
        label_images : IntArray
            The labeled images. The first dimension is interpreted as the time dimension.
        """
        self.label_images = label_images
        self.ndim = label_images.ndim - 1
        dfs = []
        for frame in range(label_images.shape[0]):
            df = pd.DataFrame(
                regionprops_table(label_images[frame], properties=["label", "bbox"])
            )
            df["frame"] = frame
            dfs.append(df)
        self.regionprops_df = pd.concat(dfs).set_index(["frame", "label"])

    def calc_overlap(
        self, frame1: Int, label1: Int, frame2: Int, label2: Int
    ) -> Tuple[Int, Float, Float, Float]:
        """Calculate the overlap properties of the labeled regions.

        Parameters
        ----------
        frame1 : Int
            the frame of the first object
        label1 : Int
            the label of the first object
        frame2 : Int
            the frame of the second object
        label2 : Int
            the label of the second object

        Returns
        -------
        overlap : float
            overlap of the labeled regions
        iou : float
            overlap over intersection of the labeled regions
        ratio_1 : float
            overlap over the area of the first object of the labeled regions
        ratio_2 : float
            overlap over the area of the second object of the labeled regions
        """
        r1 = self.regionprops_df.loc[(frame1, label1)]
        r2 = self.regionprops_df.loc[(frame2, label2)]
        bbox = self._intersect_bbox(r1, r2)
        if bbox is None:
            return 0, 0.0, 0.0, 0.0
        else:
            u_bbox = self._union_bbox(r1, r2)
            window = tuple([slice(y0, y1 + 1) for y0, y1 in u_bbox])
            b1 = self.label_images[frame1][window] == label1
            b2 = self.label_images[frame2][window] == label2
            overlap = np.sum(b1 & b2)
            union = np.sum(b1 | b2)

            return overlap, overlap / union, overlap / np.sum(b1), overlap / np.sum(b2)
