# %%
# data ... https://osf.io/ysaq2/ CC-By Attribution 4.0 International
import napari
import numpy as np


# %%
data = np.load(
    "/Volumes/share/fukai/C2C12_organized_data/FGF2/090303-C2C12P15-FGF2,BMP2_5/images.npy"
)
# %%

v = napari.Viewer()
# %%
images = data[:100:10, 300:1000, 100:800]
v.add_image(images)
# %%
labels = v.layers[-1].data
m, M = np.percentile(images, (0.01, 99.99))
images2 = (np.clip((images - m) / (M - m), 0, 1) * 255).astype(np.uint8)
np.savez_compressed("data", images=images2, labels=labels.astype(np.uint8))
