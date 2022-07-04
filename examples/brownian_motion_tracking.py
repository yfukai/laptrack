#%% # noqa:
from itertools import product

import numpy as np
from matplotlib import pyplot as plt

from laptrack import LapTrack

track_length = 100
track_count = 10
L = 10
dimension = 2
diffusion_coef = 0.1

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

#%% # noqa:
for j, pos in enumerate(brownian_poss):
    line = plt.plot(*zip(*pos), label=j)
    c = line[0].get_color()
    plt.plot(*map(lambda x: [x], init_cond[j]), "o", c=c)
plt.legend()

spots = [np.array([pos[t] for pos in brownian_poss]) for t in range(track_length)]
lt = LapTrack()
tree = lt.predict(spots)
#%% # noqa:
for edge in tree.edges():
    if (edge[0][0] + 1 != edge[1][0]) or (edge[0][1] != edge[1][1]):
        print(edge)

for i, j in product(range(track_length - 1), range(track_count)):
    if not any([edge[0] == (i, j) and edge[1] == (i + 1, j) for edge in tree.edges()]):
        print(i, j)
# %%
