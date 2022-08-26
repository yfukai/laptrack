# %% 
import napari
from laptrack import LapTrack
from glob import glob
from skimage.io import imread
from skimage.measure import regionprops_table
import numpy as np
from natsort import natsort
import pandas as pd
from laptrack import LapTrack, data_conversion

# %%
images_path = glob("/Volumes/common2/TEMPORARY/TimelapseExamples/TrackingChallenge211001/Fluo-N3DH-CE/01/*.tif")
images = np.array([imread(f) for f in images_path])
# %%
assert images_path == natsort.natsorted(images_path)
# %%
masks_path = glob("/Volumes/common2/TEMPORARY/TimelapseExamples/TrackingChallenge211001/Fluo-N3DH-CE/01_GT/TRA/*.tif")
masks = np.array([imread(f) for f in natsort.natsorted(masks_path)])
# %%
np.save("images.npy",images)
# %%
np.save("masks.npy",masks)

# %%
images = np.load("images.npy")
masks = np.load("masks.npy")
viewer=napari.Viewer()
viewer.add_image(images, scale=(1, 0.09, 0.09))
viewer.add_labels(masks, scale=(1, 0.09, 0.09))

# %%
dfs=[]
for j,m in enumerate(masks):
    r = pd.DataFrame(regionprops_table(m,properties=["label","centroid"]))
    r["frame"]=j
    dfs.append(r)

regionprops_df = pd.concat(dfs)
for k, scale in zip(["centroid-0","centroid-1","centroid-2"],(1, 0.09, 0.09)):
    regionprops_df[k]=regionprops_df[k]*scale
# %%
viewer.add_points(regionprops_df[["frame","centroid-0","centroid-1","centroid-2",]].values)

# %%
coords = data_conversion.convert_dataframe_to_coords(
    regionprops_df, ["centroid-0","centroid-1","centroid-2"])


# %%
lt = LapTrack(
    track_cost_cutoff=100**2, 
    splitting_cost_cutoff=50**2, 
    gap_closing_max_frame_count=1
)
tree = lt.predict(coords)

# %%
df, _, _ = data_conversion.convert_tree_to_dataframe(tree)
df = df.reset_index()
# %%
for j, k in enumerate(["centroid-0","centroid-1","centroid-2"]):
    df["frame"] = df["frame"].astype(np.int32)
    df["index"] = df["index"].astype(np.int32)
    def get_coords(row):
        return coords[int(row["frame"])][int(row["index"]),j]
    df[k] = df.apply(get_coords ,axis=1)

# %%
df

# %%
viewer.add_tracks(df[["track_id","frame","centroid-0","centroid-1","centroid-2"]])
# %%
viewer.scale_bar.visible = True
viewer.scale_bar.unit = "um"

# %%
