# %%
import napari
import numpy as np
import pandas as pd
from data import convert_dataframe_to_coords
from data import convert_tree_to_dataframe
from IPython.display import display
from skimage.io import imread
from skimage.measure import regionprops_table

from laptrack import LapTrack

# %%
images = imread("interactive_example_data/demo_image.tif")
labels = np.load("interactive_example_data/labels3.npy")

viewer = napari.Viewer()
viewer.add_image(images, name="images")
viewer.add_labels(labels, name="labels")

# %%
def calc_frame_regionprops(labels):  # noqa: E302
    dfs = []
    for frame in range(labels.shape[0]):
        df = pd.DataFrame(
            regionprops_table(labels[frame], properties=["label", "area", "centroid"])
        )
        df["frame"] = frame
        dfs.append(df)
    return pd.concat(dfs)


regionprops_df = calc_frame_regionprops(labels)
display(regionprops_df)

# %%

_coords = convert_dataframe_to_coords(
    regionprops_df, ["centroid-0", "centroid-1", "label"]
)
coords = [c[:, :-1] for c in _coords]
coord_labels = [c[:, -1] for c in _coords]

lt = LapTrack(splitting_cost_cutoff=20**2)
tree = lt.predict(coords)
tracked_df, _ = convert_tree_to_dataframe(tree)

# %%

_regionprops_df = regionprops_df.set_index(["frame", "label"])
for (frame, index), row in tracked_df.iterrows():
    label = coord_labels[frame][index]
    _regionprops_df.loc[(frame, label), "cell_id"] = row["cell_id"] + 1

track_label_image = np.zeros_like(labels)
for (frame, label), row in _regionprops_df.iterrows():
    track_label_image[frame][labels[frame] == label] = row["cell_id"] + 1


# %%
viewer.layers["labels"].visible = False
viewer.add_labels(track_label_image)

# %%
manual_corrected = np.load("interactive_example_data/manual_corrected.npy")
viewer.add_points(manual_corrected, name="manually_validated_tracks")

# %%
manual_corrected = viewer.layers["manually_validated_tracks"].data.astype(np.int16)
# you can also redraw the labels
new_labels = viewer.layers["track_label_image"].data
# get label values at the placed points
validated_track_labels = new_labels[tuple(manual_corrected.T)]
validated_frames = manual_corrected[:, 0]

# %%
new_regionprops_df = calc_frame_regionprops(new_labels).set_index(["frame", "label"])
new_regionprops_df.loc[
    [(f, l) for f, l in zip(validated_frames, validated_track_labels)], "cell_id"
] = -1
new_regionprops_df = new_regionprops_df.reset_index()

display(new_regionprops_df)

_new_coords = convert_dataframe_to_coords(
    new_regionprops_df[new_regionprops_df["cell_id"] != -1],
    ["centroid-0", "centroid-1", "label"],
)
new_coords = [c[:, :-1] for c in _new_coords]
new_coord_labels = [c[:, -1] for c in _new_coords]
# %%

lt = LapTrack(splitting_cost_cutoff=20**2)
new_tree = lt.predict(new_coords)
tracked_df, _ = convert_tree_to_dataframe(new_tree)

# %%

_new_regionprops_df = new_regionprops_df.copy().set_index(["frame", "label"])
for (frame, index), row in tracked_df.iterrows():
    label = new_coord_labels[frame][index]
    _new_regionprops_df.loc[(frame, label), "cell_id"] = row["cell_id"] + 1

_new_regionprops_df.loc[_new_regionprops_df["cell_id"] == -1, "cell_id"] = (
    _new_regionprops_df["cell_id"].max() + 1
)

new_track_label_image = np.zeros_like(new_labels)
for (frame, label), row in _new_regionprops_df.iterrows():
    new_track_label_image[frame][new_labels[frame] == label] = row["cell_id"] + 1


viewer.layers["track_label_image"].visible = False
viewer.add_labels(new_track_label_image)

# %%
