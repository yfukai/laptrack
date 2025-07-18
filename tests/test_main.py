"""Test cases for the __main__ module."""
import os
from pathlib import Path

import numpy as np
import zarr
from skimage.io import imsave

from laptrack import __main__
from laptrack import LapTrack
from laptrack.datasets import cell_segmentation
from laptrack.datasets import HL60_3D_synthesized


def test_read_labels(tmp_path):
    _, labels = cell_segmentation()
    _, labels2 = HL60_3D_synthesized()
    os.chdir(tmp_path)
    np.save("labels.npy", labels)
    np.save("labels2.npy", labels2)
    imsave("labels.tif", labels, check_contrast=False)
    imsave("labels2.tif", labels2, check_contrast=False)
    zarr.save("labels.zarr", labels)
    zarr.save("labels2.zarr", labels2)
    os.makedirs("labels", exist_ok=True)
    os.makedirs("labels2", exist_ok=True)
    for frame, l in enumerate(labels):
        imsave(f"labels/labels_{frame:04d}.tif", l, check_contrast=False)
    for frame, l in enumerate(labels2):
        imsave(f"labels2/labels2_{frame:04d}.tif", l, check_contrast=False)
    assert (labels == __main__._read_image("labels.tif")).all()
    assert (labels2 == __main__._read_image("labels2.tif")).all()
    assert (labels == __main__._read_image("labels.zarr")).all()
    assert (labels2 == __main__._read_image("labels2.zarr")).all()
    assert (labels == __main__._read_image("labels.npy")).all()
    assert (labels2 == __main__._read_image("labels2.npy")).all()
    assert (labels == __main__._read_image("labels/")).all()
    assert (labels2 == __main__._read_image("labels2/")).all()


def test_tap_configure():
    args = __main__._TrackArgs().parse_args(
        "--coordinate_cols position_x position_y --csv_path test.csv --output_path test.zarr --metric sqeuclidean --cutoff 255".split()
    )
    lt_kwargs = {name: getattr(args, name) for name in LapTrack.model_fields}
    lt = LapTrack(**lt_kwargs)
    assert args.csv_path == Path("test.csv")
    assert args.output_path == Path("test.zarr")
    assert args.coordinate_cols == ["position_x", "position_y"]
    assert args.frame_col == "frame"
    assert lt.metric == "sqeuclidean"
    assert lt.cutoff == 255
    assert not lt.splitting_cutoff
    print(lt)
