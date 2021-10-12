#%% # noqa:
import napari
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.feature import blob_log

from laptrack import laptrack

#%% # noqa:
viewer = napari.Viewer()
#%% # noqa:
track_length = 100
track_count = 10
L = 10
pixel_size = 0.1
diameter = 1
dimension = 2
diffusion_coef = 0.01

#%% # noqa:

init_cond = np.random.rand(track_count, dimension) * L

brownian_poss = np.array(
    [
        init_cond[i]
        + np.concatenate(
            [
                [np.zeros(dimension)],
                np.cumsum(
                    np.sqrt(diffusion_coef)
                    * np.random.normal(size=(track_length, dimension)),
                    axis=0,
                ),
            ]
        )
        for i in range(track_count)
    ]
)


# %%
L2 = int(L / pixel_size)
images = np.zeros((track_length, L2, L2))
xx, yy = np.mgrid[:L2, :L2]
for i in range(track_length):
    for pos in brownian_poss[:, i]:
        pos2 = pos / pixel_size
        images[i] += np.exp(
            -((xx - pos2[0]) ** 2 + (yy - pos2[1]) ** 2) / 2 / diameter ** 2
        )
# %%

viewer.add_image(images)
# %%
spots = []
for j, image in enumerate(images):
    _spots = blob_log(image, min_sigma=2, max_sigma=10, num_sigma=3)
    spots.append(np.hstack([np.ones((_spots.shape[0], 1)) * j, _spots[:, :-1]]))
spots = np.vstack(spots).astype(np.int32)
# %%
viewer.add_points(spots, size=3, edge_color="yellow", face_color="#ffffff00")

# %%
spots_for_tracking = [spots[spots[:, 0] == j][:, 1:] for j in range(track_length)]
track_tree = laptrack(spots_for_tracking)

tracks = []
for i, segment in enumerate(nx.connected_components(track_tree)):
    _tracks = []
    for spot in segment:
        frame = spot[0]
        index = spot[1]
        _tracks.append([frame, *spots_for_tracking[frame][index]])
    tracks.append(np.hstack([np.ones((len(_tracks), 1)) * i, _tracks]))
tracks = np.vstack(tracks).astype(np.int32)
# %%
viewer.add_tracks(tracks)
# %%

for ind in np.unique(tracks[:, 0]):
    data = tracks[tracks[:, 0] == ind]
    ind2 = np.argsort(data[:, 1])
    plt.plot(data[ind2, 1], data[ind2, 3], "-")

# %%
extracted_spots = spots[
    (spots[:, 0] > 21) & (spots[:, 0] < 24) & (spots[:, 2] > 20) & (spots[:, 2] < 50), :
]

pd.DataFrame(extracted_spots, columns=["t", "x", "y"]).to_csv(
    "test_spots.csv", index=False
)

# %%

plt.plot(extracted_spots[:, 0], extracted_spots[:, 1], ".")
plt.plot(extracted_spots[:, 0], extracted_spots[:, 2], ".")
plt.show()

extracted_spots_for_tracking = [
    extracted_spots[extracted_spots[:, 0] == j, 1:]
    for j in np.sort(np.unique(extracted_spots[:, 0]))
]

test_track = laptrack(extracted_spots_for_tracking, gap_closing_cost_cutoff=False)
for edge in test_track.edges():
    _e = np.array(edge)
    pos = [extracted_spots_for_tracking[frame][ind] for frame, ind in edge]
    plt.plot(_e[:, 0], pos)
plt.show()

segments = nx.connected_components(test_track)

test_tracks = []
for i, segment in enumerate(segments):
    _tracks = []
    for spot in segment:
        frame = spot[0]
        index = spot[1]
        _tracks.append([frame, *spots_for_tracking[frame][index]])
    test_tracks.append(np.hstack([np.ones((len(_tracks), 1)) * i, _tracks]))
test_tracks = np.vstack(test_tracks).astype(np.int32)

for ind in np.unique(test_tracks[:, 0]):
    data = test_tracks[test_tracks[:, 0] == ind]
    ind2 = np.argsort(data[:, 1])
    plt.plot(data[ind2, 1], data[ind2, 2], "-")

# %%
