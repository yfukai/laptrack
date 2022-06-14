from os import path

import pandas as pd

from laptrack import laptrack

script_path = path.dirname(path.realpath(__file__))
filename = "../tests/data/trackmate_tracks_with_splitting_spots.csv"
spot_file_path = path.join(script_path, filename)
spots_df = pd.read_csv(spot_file_path)
frame_max = spots_df["frame"].max()
coords = []
spot_ids = []
for i in range(frame_max):
    df = spots_df[spots_df["frame"] == i]
    coords.append(df[["position_x", "position_y"]].values)
    spot_ids.append(df["id"].values)

print(coords)
# point coordinates at each frame
# [array([[116.28953441, 116.35973216],
#         [ 63.97088847,   4.01865591]]),
#  array([[ 64.12021924,   6.17349359],
#         [116.18952078, 116.27853574]]),
#  ...
# ]

max_distance = 15
track_tree = laptrack(
    coords,
    track_dist_metric="sqeuclidean",
    splitting_dist_metric="sqeuclidean",
    track_cost_cutoff=max_distance**2,
    splitting_cost_cutoff=max_distance**2,
)

for edge in track_tree.edges():
    print(edge)


# connection between spots, represented as ((frame1,index1), (frame2,index2)) ...
# for example, ((0, 0), (1, 1)) means the connection between
# [116.28953441, 116.35973216] and [116.18952078, 116.27853574] in this example.
# ((0, 0), (1, 1))
# ((0, 1), (1, 0))
# ((1, 0), (2, 0))
# ...
