# %%
import napari
import networkx as nx
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.measure import regionprops_table

from laptrack import LapTrack

#%% # noqa:
images = imread("interactive_example_data/demo_image.tif")
labels = np.load("interactive_example_data/labels3.npy")
for f in range(labels.shape[0]):
    print(np.unique(labels[f]))
    label_correspondence = {
        i + 1: label for i, label in enumerate(np.unique(labels[f])) if label != 0
    }
    tmp = labels[f].copy()
    labels[f] = np.zeros_like(labels[f])
    for k, v in label_correspondence.items():
        labels[f][tmp == v] = k

# %%
viewer = napari.Viewer()
viewer.add_image(images, name="images")
viewer.add_labels(labels, name="labels")
# %%
labels2 = viewer.layers["labels"].data
np.save("interactive_example_data/labels3.npy", labels2)

# %%
dfs = []
for frame in range(labels.shape[0]):
    df = pd.DataFrame(
        regionprops_table(labels[frame], properties=["label", "area", "centroid"])
    )
    df["frame"] = frame
    dfs.append(df)
regionprops_df = pd.concat(dfs)
regionprops_df
# %%
grps = list(regionprops_df.groupby("frame", sort=True))
assert np.array_equal(np.arange(10), [g[0] for g in grps])
coords = [grp[["centroid-0", "centroid-1"]].values for frame, grp in grps]
coord_labels = [grp["label"].values for frame, grp in grps]
lt = LapTrack(splitting_cost_cutoff=20**2)
tree = lt.predict(coords)
# %%


def convert_tree_to_dataframe(tree):
    df_data = []
    node_values = np.array(list(tree.nodes))
    frames = np.unique(node_values[:, 0])
    for frame in frames:
        indices = node_values[node_values[:, 0] == frame, 1]
        df_data.append(
            pd.DataFrame(
                {
                    "frame": [frame] * len(indices),
                    "index": indices,
                }
            )
        )
    df = pd.concat(df_data).set_index(["frame", "index"])
    connected_components = list(nx.connected_components(tree))
    for track_id, nodes in enumerate(connected_components):
        for (frame, index) in nodes:
            df.loc[(frame, index), "clone_id"] = track_id
            tree.nodes[(frame, index)]["clone_id"] = track_id
    tree2 = tree.copy()
    for node in tree.nodes:
        frame0, index0 = node
        neighbors = list(tree.neighbors(node))
        children = [(frame, index) for (frame, index) in neighbors if frame > frame0]
        parents = [(frame, index) for (frame, index) in neighbors if frame < frame0]
        assert len(children) + len(parents) == len(neighbors)
        if len(children) > 1:
            for child in children:
                tree2.remove_edge(node, child)
        if len(parents) > 1:
            for parent in parents:
                tree2.remove_edge(node, parent)

    connected_components = list(nx.connected_components(tree2))
    for track_id, nodes in enumerate(connected_components):
        for (frame, index) in nodes:
            df.loc[(frame, index), "cell_id"] = track_id
            tree.nodes[(frame, index)]["cell_id"] = track_id

    for k in ["clone_id", "cell_id"]:
        df[k] = df[k].astype(int)
    return df, None


df, _ = convert_tree_to_dataframe(tree)

df
# %%
track_label_image = np.zeros_like(labels)
for (frame, index), row in df.iterrows():
    label = coord_labels[frame][index]
    df.loc[(frame, index), "label"] = label
    track_label_image[frame][labels[frame] == label] = row["cell_id"] + 1


# %%
viewer.layers["labels"].visible = False
viewer.add_labels(track_label_image)

# %%
manual_corrected = viewer.layers["Points"].data
np.save("interactive_example_data/manual_corrected.npy", manual_corrected)

# %%
viewer.add_points(manual_corrected, name="manually_validated_tracks")
# %%
manual_corrected = viewer.layers["manually_validated_tracks"].data.astype(np.int16)
labels = viewer.layers["labels"].data
validated_track_labels = track_label_image[tuple(manual_corrected.T)]


# %%
