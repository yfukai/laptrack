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
        "hash": None,  # "sha256:44c78a369da8aaa53e6a53c10766d0f4451b54e9e8c84ff27bf47ac2cd2037a6",
    },
    "bright_brownian_particles": {
        "filename": "bright_spots_data/brownian_particles_images_with_noise_small.npz",
        "hash": None,  # "sha256:1ae73102d586a6ccf2d603c81316359bf35ea490ddf146ac37ac2e671ce45111",
    },
    "cell_segmentation": {
        "filename": "cell_segmentation_data/data_small.npz",
        "hash": None,  # "sha256:48040529876dcc9142c5b1b0448c8773c1657d2b14a2807b32ea911e8362f3e9",
    },
    "mouse_epidermis": {
        "filename": "overlap_tracking_data/labels.npy",
        "hash": None,  # "sha256:b04d0f38a99ce09e5202bf844cb768eca3919f6728c36b01b94c7fab9c2344b2",
    },
}


POOCH = pooch.create(
    path=pooch.os_cache("laptrack"),
    # Use the Zenodo DOI
    base_url="https://raw.githubusercontent.com/yfukai/laptrack/9819adef1490b1fd7270e252c589933fe9b435cb/docs/examples/",
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
    if path.splitext(file_name)[1] == ".csv":
        return pd.read_csv(test_file_path)
    else:
        return np.load(test_file_path)


def simple_tracks() -> pd.DataFrame:
    """Return the "simple tracks" dataset.

    Returns
    -------
    data: pd.DataFrame
        The result data.
    """
    return fetch("simple_tracks")


def bright_brownian_particles() -> FloatArray:
    """Return the "bright Brownian particles" dataset.

    Returns
    -------
    data: pd.DataFrame
        The result data.
    """
    return fetch("bright_brownian_particles")["images"]


def cell_segmentation() -> Tuple[FloatArray, IntArray]:
    """Return the "cell segmentation" dataset.

    Data source: https://osf.io/ysaq2/ CC-By Attribution 4.0 International

    Returns
    -------
    data: pd.DataFrame
        The result data.
    """
    d = fetch("cell_segmentation")
    return d["images"], d["labels"]


def mouse_epidermis() -> IntArray:
    """Return the "mouse epidermis" dataset.

    Data source: cropping `segmentation.npy` in
    https://github.com/NoneqPhysLivingMatterLab/cell_interaction_gnn.

    Returns
    -------
    data: pd.DataFrame
        The result data.
    """
    return fetch("mouse_epidermis")
