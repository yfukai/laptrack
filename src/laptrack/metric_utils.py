"""Utilities for metric calculation."""
import sys

if sys.version_info >= (3, 9):
    from functools import cache
else:
    from functools import lru_cache

    cache = lru_cache(maxsize=None)

from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from sklearn.metrics import confusion_matrix

from ._typing_utils import Float
from ._typing_utils import Int
from ._typing_utils import IntArray


class LabelOverlap:
    """Utility object to calculate overlap of segmentation labels between frames."""

    def __init__(self, label_images: Union[IntArray, List[IntArray]]):
        """Summarise the segmentation properties and initialize the object.

        Parameters
        ----------
        label_images : Union[IntArray,List[IntArray]]
            The labeled images. The first dimension is interpreted as the frame dimension.
        """
        if not isinstance(label_images, np.ndarray):
            label_images = np.array(label_images)
        if label_images.ndim < 3:
            raise ValueError("label_images dimension must be >=3.")
        self.label_images = label_images
        self.ndim = label_images.ndim - 1
        self.unique_labels = []

        # TODO parallelize
        # Calculate unique labels for each frame
        dfs = []
        for frame in range(label_images.shape[0]):
            unique_labels = np.unique(label_images[frame])
            self.unique_labels.append(unique_labels)
            dfs.append(
                pd.DataFrame(dict(label=np.trim_zeros(unique_labels))).assign(
                    frame=frame
                )
            )
        self.frame_label_df = pd.concat(dfs)[["frame", "label"]]

    @cache
    def _overlap_matrix(self, frame1, frame2):
        label1 = self.label_images[frame1]
        label2 = self.label_images[frame2]
        unique_labels1 = self.unique_labels[frame1]
        unique_labels2 = self.unique_labels[frame2]
        unique_labels_combined = np.union1d(unique_labels1, unique_labels2)
        unique_labels_combined_dict = dict(
            zip(unique_labels_combined, np.arange(len(unique_labels_combined)))
        )
        overlap_matrix = confusion_matrix(
            label1.ravel(), label2.ravel(), labels=unique_labels_combined
        )
        return overlap_matrix, unique_labels_combined_dict

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
            overlap (intersection) of the labeled regions
        iou : float
            overlap over the union of the labeled regions
        ratio_1 : float
            overlap over the area of the first object of the labeled regions
        ratio_2 : float
            overlap over the area of the second object of the labeled regions
        """
        overlap_matrix, unique_labels_combined_dict = self._overlap_matrix(
            frame1, frame2
        )
        index_1 = unique_labels_combined_dict[label1]
        index_2 = unique_labels_combined_dict[label2]

        overlap = overlap_matrix[index_1, index_2]
        if overlap == 0:
            return 0, 0.0, 0.0, 0.0
        b1_sum = np.sum(overlap_matrix[index_1, :])
        b2_sum = np.sum(overlap_matrix[:, index_2])
        union = b1_sum + b2_sum - overlap
        return overlap, overlap / union, overlap / b1_sum, overlap / b2_sum

    def __del__(self):
        """Destructor."""
        self._overlap_matrix.cache_clear()


class LabelOverlapOld:
    """Utility object to calculate overlap of segmentation labels between frames (Old implementation using regionoprops)."""

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

    def __init__(self, label_images: Union[IntArray, List[IntArray]]):
        """Summarise the segmentation properties and initialize the object.

        Parameters
        ----------
        label_images : Union[IntArray,List[IntArray]]
            The labeled images. The first dimension is interpreted as the frame dimension.
        """
        if not isinstance(label_images, np.ndarray):
            label_images = np.array(label_images)
        if label_images.ndim < 3:
            raise ValueError("label_images dimension must be >=3.")
        self.label_images = label_images
        self.ndim = label_images.ndim - 1
        dfs = []
        for frame in range(label_images.shape[0]):
            df = pd.DataFrame(
                regionprops_table(label_images[frame], properties=["label", "bbox"])
            )
            df["frame"] = frame
            dfs.append(df)
        df = pd.concat(dfs)
        self.regionprops_df = df.set_index(["frame", "label"])
        self.frame_label_df = df[["frame", "label"]]

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
