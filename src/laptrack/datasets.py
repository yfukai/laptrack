"""Datasets used for testing."""
from os import path
from typing import Tuple

import numpy as np
import pandas as pd
import pooch

from ._typing_utils import FloatArray
from ._typing_utils import IntArray

TEST_DATA_PROPS = {
    "simple_tracks": {
        "filename": "sample_data.csv",
        "hash": "md5:c3029c5e93157aa49a6716e864e979bd",
    },
    "bright_brownian_particles": {
        "filename": "bright_spots_data/brownian_particles_images_with_noise_small.npz",
        "hash": "md5:a93742566635fbc5dde7f3a653e638c1",
    },
    "cell_segmentation": {
        "filename": "cell_segmentation_data/data_small.npz",
        "hash": "md5:be37bfb4da12f47b6e3146d22fb80290",
    },
    "mouse_epidermis": {
        "filename": "overlap_tracking_data/labels.npy",
        "hash": "md5:a1b12b0d08c894b804d5440010cef77e",
    },
    "HL60_3D_synthesized": {
        "filename": "3D_tracking_data/HL60_3D_synthesized_data.npz",
        "hash": "md5:3d6536400398464d9b0bdc88878115de",
    },
}

POOCH = pooch.create(
    path=pooch.os_cache("laptrack"),
    # Use the Zenodo DOI
    base_url="https://raw.githubusercontent.com/yfukai/laptrack/97943e8b61a5fa6e9cdabf2968037c4b1f8cbf32/docs/examples/",
    registry={v["filename"]: v["hash"] for v in TEST_DATA_PROPS.values()},
)


def fetch(data_name: str):
    """Fetch a sample dataset from GitHub.

    Parameters
    ----------
    data_name: str
        The name of the dataset. Must be one of ["simple_tracks",
        "bright_brownian_particles", "cell_segmentation", "mouse_epidermis"].

    Returns
    -------
    data: Union[ndarray,pd.DataFrame]
        The result data.

    Raises
    ------
        ValueError: If the dataset name is not one of the allowed values.
    """
    if data_name not in TEST_DATA_PROPS.keys():
        raise ValueError(f"{data_name} is not a valid test data name")
    file_name = TEST_DATA_PROPS[data_name]["filename"]
    test_file_path = POOCH.fetch(file_name)
    if path.splitext(test_file_path)[1] == ".csv":
        return pd.read_csv(test_file_path)
    else:
        return np.load(test_file_path)


def simple_tracks() -> pd.DataFrame:
    """Return the "simple tracks" dataset.

    Returns
    -------
    data: pd.DataFrame
        The dataset.
    """
    return fetch("simple_tracks")


def bright_brownian_particles() -> FloatArray:
    """Return the "bright Brownian particles" dataset.

    Returns
    -------
    data: FloatArray
        The images.
    """
    return fetch("bright_brownian_particles")["images"]


def cell_segmentation() -> Tuple[FloatArray, IntArray]:
    """Return the "cell segmentation" dataset.

    Data source: https://osf.io/ysaq2/ CC-By Attribution 4.0 International

    Returns
    -------
    data: Tuple[FloatArray, IntArray]
        The images and labels.
    """
    d = fetch("cell_segmentation")
    return d["images"], d["labels"]


def mouse_epidermis() -> IntArray:
    """Return the "mouse epidermis" dataset.

    Data source: cropping `segmentation.npy` in
    https://github.com/NoneqPhysLivingMatterLab/cell_interaction_gnn.

    Returns
    -------
    data: IntArray
        The labels.
    """
    return fetch("mouse_epidermis")


def HL60_3D_synthesized() -> Tuple[FloatArray, IntArray]:
    """Return the "HL60 3D synthesized" dataset.

    Data source: rescaling dataset 1 in
    https://bbbc.broadinstitute.org/BBBC050
    Image set BBBC050 [Tokuoka, Yuta, et al. npj Syst Biol Appl 6, 32 (2020)],
    downloaded from the Broad Bioimage Benchmark Collection
    [Ljosa et al., Nature Methods, 2012].
    The images and ground truth are licensed under
    a Creative Commons Attribution 3.0 Unported License
    (Commercial use allowed) by Akira Funahashi.

    Returns
    -------
    data: Tuple[FloatArray, IntArray]
        The images and labels.
    """
    data = fetch("HL60_3D_synthesized")
    return data["images"], data["labels"]
