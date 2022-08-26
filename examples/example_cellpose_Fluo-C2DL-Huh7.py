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
from cellpose import models, core

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# %%
images_path = glob("/Volumes/common2/TEMPORARY/TimelapseExamples/TrackingChallenge211001/train2D/DIC-C2DH-HeLa/01/*.tif")
images = np.array([imread(f) for f in natsort.natsorted(images_path)])
# %%
viewer=napari.Viewer()
viewer.add_image(images)

# %%
cp=models.Cellpose()
masks, flows, styles, diams= cp.eval(list(images[:1]), diameter=100, progress=True)

# %%
viewer.add_labels(masks)

# %%
dfs=[]
for j,m in enumerate(masks):
    r = pd.DataFrame(regionprops_table(m,properties=["label","centroid"]))
    r["frame"]=j
    dfs.append(r)

regionprops_df = pd.concat(dfs)
# %%
viewer.add_points(regionprops_df[["frame","centroid-0","centroid-1","centroid-2",]].values, scale=(1,1, 0.09, 0.09))

# %%
coords = data_conversion.convert_dataframe_to_coords(regionprops_df, ["centroid-0","centroid-1","centroid-2"])


# %%
lt = LapTrack(track_cost_cutoff=100**2, splitting_cost_cutoff=100**2)
tree = lt.predict(coords)

# %%
df, _, _ = data_conversion.convert_tree_to_dataframe(tree)
df = df.reset_index()
# %%
for j, (k, scale) in enumerate(zip(["centroid-0","centroid-1","centroid-2"],[1,0.09,0.09])):
    df["frame"] = df["frame"].astype(np.int32)
    df["index"] = df["index"].astype(np.int32)
    def get_coords(row):
        return coords[int(row["frame"])][int(row["index"]),j]* scale
    df[k] = df.apply(get_coords ,axis=1)

# %%
df

# %%
viewer.add_tracks(df[["track_id","frame","centroid-0","centroid-1","centroid-2"]])
# %%
viewer.scale_bar.visible = True

# %%
