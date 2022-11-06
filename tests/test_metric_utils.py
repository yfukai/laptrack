from itertools import product

import numpy as np

from laptrack.metric_utils import LabelOverlap


def test_label_overlap() -> None:
    labels = np.array(
        [
            [[[0, 1, 1, 1, 0], [0, 1, 2, 2, 2]], [[0, 1, 2, 2, 2], [3, 3, 3, 1, 0]]],
            [[[0, 1, 1, 1, 2], [0, 4, 1, 2, 2]], [[0, 4, 4, 4, 4], [0, 4, 4, 4, 4]]],
            [[[0, 1, 1, 1, 0], [5, 5, 5, 5, 5]], [[0, 1, 1, 1, 0], [0, 1, 1, 1, 0]]],
        ]
    )
    labelss = [labels, list(labels)]

    for labels in labelss:
        lo = LabelOverlap(labels)
        frame_labels = [np.unique(label) for label in labels]
        frame_labels = [x[x > 0] for x in frame_labels]
        for f1, f2 in [(0, 0), (0, 1), (1, 2)]:
            for l1, l2 in product(frame_labels[f1], frame_labels[f2]):
                b1 = labels[f1] == l1
                b2 = labels[f2] == l2

                intersect = np.sum(b1 & b2)
                union = np.sum(b1 | b2)
                r1 = np.sum(b1)
                r2 = np.sum(b2)
                res = lo.calc_overlap(f1, l1, f2, l2)
                assert (
                    intersect,
                    (intersect / union),
                    (intersect / r1),
                    (intersect / r2),
                ) == res
