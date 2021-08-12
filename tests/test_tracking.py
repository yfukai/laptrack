"""Test cases for the tracking."""
from os import path
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from laptrack import track

DEFAULT_PARAMS=dict(
    track_distance_cutoff = 15,
    track_start_cost= 30, 
    track_end_cost= 30, 
    gap_closing_cutoff= 15,
    gap_closing_max_frame_count= 2,
    splitting_cutoff= False,
    no_splitting_cost= 30,
    merging_cutoff= False,
    no_merging_cost= 30, 
)

FILENAME_SUFFIX_PARAMS=[
    ("without_gap_closing",{
        **DEFAULT_PARAMS,
        "gap_closing_cutoff":False,
        "splitting_cutoff":False,
        "merging_cutoff":False,
        }),
#    ("with_gap_closing",{
#       **DEFAULT_PARAMS,
#       "splitting_cutoff":False,
#       "merging_cutoff":False,
#       }),
#    ("with_splitting",{
#       **DEFAULT_PARAMS,
#       "merging_cutoff":False,
#       }),
#    ("with_merging",{
#        **DEFAULT_PARAMS,
#        "splitting_cutoff":False,
#        }),
]

def test_tracking(shared_datadir: str) -> None:
    for filename_suffix, params in FILENAME_SUFFIX_PARAMS:
        filename=path.join(shared_datadir,
                            f"trackmate_tracks_{filename_suffix}")
        spots_df = pd.read_csv(filename+"_spots.csv")
        frame_max = spots_df["frame"].max()
        coords=[]
        spot_ids=[]
        for i in range(frame_max):
            df=spots_df[spots_df["frame"]==i]
            coords.append(
                df[["position_x","position_y"]].values
            )
            spot_ids.append(
                df["id"].values
            )
        track_tree=track(coords,**params)
        edges_df = pd.read_csv(filename+"_edges.csv")


