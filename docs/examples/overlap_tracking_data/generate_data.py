# %%
import napari
import numpy as np

# from https://github.com/NoneqPhysLivingMatterLab/cell_interaction_gnn/blob/main/data/epidermis/paw/W-R1/segmentation.npy
orig_labels = np.load("/Users/fukai/Downloads/segmentation.npy", allow_pickle=True)
# %%
labels = orig_labels[:, 128 : 256 + 128, 128 : 256 + 128]
labels_new = np.zeros_like(labels)

for frame in range(len(labels)):
    unique_labels = np.unique(labels[frame])
    unique_labels = unique_labels[unique_labels > 0]
    for j, l in enumerate(unique_labels):
        labels_new[frame][labels[frame] == l] = j + 1

assert labels_new.max() < 256
# %%

viewer = napari.Viewer()
viewer.add_labels(labels_new)
# %%

np.save("labels.npy", labels_new[:5].astype(np.uint8))
# %%

viewer.add_labels(labels_new == 2)

# %%
for label in labels_new:
    print(np.unique(label))

# %%
