# %%
from os import path

import pandas as pd

from laptrack import LapTrack

script_path = path.dirname(path.realpath(__file__))
filename = "../tests/data/trackmate_tracks_with_splitting_spots.csv"
spot_file_path = path.join(script_path, filename)
spots_df = pd.read_csv(spot_file_path, index_col=0)

max_distance = 15
lt = LapTrack(
    track_dist_metric="sqeuclidean",
    splitting_dist_metric="sqeuclidean",
    track_cost_cutoff=max_distance**2,
    splitting_cost_cutoff=max_distance**2,
    merging_cost_cutoff=max_distance**2,
)

##### example 1: pd.DataFrame -> pd.DataFrame #####

track_df, split_df, merge_df = lt.predict_dataframe(
    spots_df,
    ["position_x", "position_y"],
    only_coordinate_cols=False,
)

print(
    track_df[
        [
            "position_x",
            "position_y",
            "track_id",
            "tree_id",
        ]
    ]
)

#              position_x  position_y  tree_id  track_id
# frame index
# 0     0      116.289534  116.359732        0         0
#       1       63.970888    4.018656        1         1
# 1     0       64.120219    6.173494        1         1
#       1      116.189521  116.278536        0         0
# 2     0       64.088872    9.962219        1         1
# ...                 ...         ...      ...       ...

# The original dataframe with additional columns "track_id" and "tree_id".
# The track_id is a unique id for each track segments without branches.
# A new id is assigned when a splitting and merging occured. The tree_id
# is a unique id for each "clonal" tracks sharing the same ancestor.

print(split_df)

#    parent_track_id  child_track_id
# 0                1               2
# 1                1               3
# 2                3               5
# 3                3               4

# The dataframe for splitting events with the following columns:
# - "parent_track_id" : the track id of the parent
# - "child_track_id" : the track id of the parent

print(merge_df)

#    parent_track_id  child_track_id
# 0                5               5
# 1                5               5
# 2                2               2
# 3                2               2

# The dataframe for merging events with the following columns:
# - "parent_track_id" : the track id of the parent
# - "child_track_id" : the track id of the parent

##### example 2: coordinates -> nx.DiGraph #####

frame_max = spots_df["frame"].max()
coords = []
spot_ids = []
for i in range(frame_max):
    df = spots_df[spots_df["frame"] == i]
    coords.append(df[["position_x", "position_y"]].values)
    spot_ids.append(df["id"].values)

print(coords)
# [array([[116.28953441, 116.35973216],
#         [ 63.97088847,   4.01865591]]),
#  array([[ 64.12021924,   6.17349359],
#         [116.18952078, 116.27853574]]),
#  ...
# ]

# point coordinates at each frame

track_tree = lt.predict(
    coords,
)

for edge in track_tree.edges():
    print(edge)

# ((0, 0), (1, 1))
# ((0, 1), (1, 0))
# ((1, 0), (2, 0))
# ...

# connection between spots, represented as ((frame1,index1), (frame2,index2)) ...
# for example, ((0, 0), (1, 1)) means the connection between
# [116.28953441, 116.35973216] and [116.18952078, 116.27853574] in this example.


# %%
