# %%
# %pip install jax==0.3.15 jaxlib==0.3.15 jax-md seaborn
# %%
import os

import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import napari
import numpy as np
import seaborn as sns
from jax import lax
from jax import random
from jax.config import config
from jax_md import energy
from jax_md import quantity
from jax_md import simulate
from jax_md import space

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")
sns.set_style(style="white")


def format_plot(x, y):
    plt.xlabel(x, fontsize=20)
    plt.ylabel(y, fontsize=20)


def finalize_plot(shape=(1, 1)):
    plt.gcf().set_size_inches(
        shape[0] * 1.5 * plt.gcf().get_size_inches()[1],
        shape[1] * 1.5 * plt.gcf().get_size_inches()[1],
    )
    plt.tight_layout()


# %%
N = 400
dimension = 2
box_size = quantity.box_size_at_number_density(N, 1, dimension)
dt = 1e-3
displacement, shift = space.periodic(box_size)
kT = 0.025

# %%
key = random.PRNGKey(0)
key, split = random.split(key)

initial_positions = box_size * random.uniform(split, (N, dimension), dtype=jnp.float64)
species = jnp.array([0] * (N // 2) + [1] * (N - N // 2))
sigmas = jnp.array([[1.0, 1.2], [1.2, 1.4]])

energy_fn = energy.soft_sphere_pair(displacement, sigma=sigmas, species=species)
init_fn, apply_fn = simulate.brownian(energy_fn, shift, dt, kT)
state = init_fn(key, initial_positions)
# %%
# %%
plt.rcParams["font.family"] = ""
write_count = 100
write_every = 10


def simulate_brownian(write_every):
    def step_fn(i, state_log):
        state, log = state_log
        log["position"] = lax.cond(
            i % write_every == 0,
            lambda p: p.at[i // write_every].set(state.position),
            lambda p: p,
            log["position"],
        )
        state = apply_fn(state, kT=kT)
        return state, log

    os.makedirs("write_every_{write_every}", exist_ok=True)
    steps = write_every * write_count
    log = {"position": jnp.zeros((steps // write_every,) + initial_positions.shape)}
    _, log = lax.fori_loop(0, steps, step_fn, (state, log))
    np.save("brownian_particles", np.array(log["position"]))

    fig = plt.figure()
    ims = []
    for t in range(len(log["position"])):
        im = plt.plot(log["position"][t, :, 0], log["position"][t, :, 1], ".b")
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save("brownian_particles.gif", fps=4)


print(write_every)
simulate_brownian(write_every)


# %%

viewer = napari.Viewer()
# %%
pixel_size = 0.1
diameter = 1
L = 20
poss = np.load("brownian_particles.npy")
# %%

L2 = int(L / pixel_size)
images = np.zeros((poss.shape[0], L2, L2))
xx, yy = np.mgrid[:L2, :L2]
for t, pos in enumerate(poss):
    for p in pos:
        pos2 = p / pixel_size
        images[t] += np.exp(
            -((xx - pos2[0]) ** 2 + (yy - pos2[1]) ** 2) / 2 / diameter**2
        )

# %%
images2 = images[50::5]
viewer.add_image(images2)
np.save("brownian_particles_images.npy", images2)
images3 = images2 + np.random.normal(scale=0.05, size=images2.shape)
viewer.add_image(images3)
np.savez_compressed(
    "brownian_particles_images_with_noise.npz", images=images3[:, :50, :50]
)

# %%
images3 = np.load("brownian_particles_images_with_noise.npz")["images"]
np.savez_compressed(
    "brownian_particles_images_with_noise_small.npz", images=images3[:, :80, :80]
)

# %%
