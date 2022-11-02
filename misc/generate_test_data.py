# %%
import napari
import numpy as np
import pandas as pd

viewer = napari.Viewer()
# %%
im = np.zeros((20, 10, 10))
viewer.add_image(im)
# %%
points = viewer.layers[1].data
# %%
points
# %%
df = pd.DataFrame(points, columns=["frame", "position_x", "position_y"])
df["frame"] = df["frame"].astype(int)

# %%
df.to_csv("sample_data.csv", index=False)
# %%
