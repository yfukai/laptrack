# %%
from pathlib import Path

import napari
import numpy as np
from skimage.io import imread
from skimage.transform import rescale
from tqdm import tqdm

# %%

image_dir = Path("~/Downloads/BBBC035_v1_dataset/01/").expanduser()
mask_dir = Path("~/Downloads/BBBC035_v1_DatasetGroundTruth/01_GT/SEG").expanduser()

image_files = list(image_dir.glob("t???.tif"))
mask_files = list(mask_dir.glob("man_seg???.tif"))
ts = []
# zs = []
for f in image_files:
    ts.append(int(f.name[1:4]))
#    zs.append(int(f.name[14:17]))
ts = np.unique(ts)
print(ts)
# zs = np.unique(zs)

testimg = imread(image_files[0])
testmask = imread(mask_files[0])
print(testimg.shape, testimg.dtype)
# %%
img = np.empty((len(ts), *testimg.shape), dtype=testimg.dtype)
mask = np.empty((len(ts), *testimg.shape), dtype=testmask.dtype)

for j, t in tqdm(enumerate(ts)):
    img[j] = imread(image_dir / f"t{t:03d}.tif")
    mask[j] = imread(mask_dir / f"man_seg{t:03d}.tif")

# %%
viewer = napari.Viewer()
viewer.add_image(img, name="img")
viewer.add_labels(mask, name="mask")

# %%
scale = 0.125
max_time = 75

img2 = rescale(
    img[:max_time], (1, scale * 0.200 / 0.125, scale, scale), preserve_range=True
)
mask2 = rescale(
    mask[:max_time],
    (1, scale * 0.200 / 0.125, scale, scale),
    order=0,
    preserve_range=True,
    anti_aliasing=False,
)


mask2 = mask2.astype(np.uint8)
# %%
viewer.add_image(img2, name="img2")
viewer.add_labels(mask2, name="mask2")
np.savez("HL60_3D_synthesized.npz", images=img2, labels=mask2)
# %%
