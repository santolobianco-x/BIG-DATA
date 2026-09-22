import random
import numpy as np

from grid_network import GridNetwork


def build_watts_strogatz(n, num_links_per_node=1, seed=None):
    """
    Costruisce una rete Watts-Strogatz

    I link a lunga distanza sono scelti uniformemente a caso
    """

    if seed is not None:
        random.seed(seed)


    net = GridNetwork(n)

    for i in range(n):
        for j in range(n):

            src = (i, j)

            for _ in range(num_links_per_node):

                while True:

                    dst = (
                        random.randint(0, n-1),
                        random.randint(0, n-1)
                    )

                    if dst != src:
                        break

                net.add_long_link(src,dst)

    return net

def build_kleinberg(
    n,
    num_links_per_node=1,
    alpha=2.0,
    seed=None
):
    """
    Costruisce una rete Kleinberg.

    La probabilità di scegliere v partendo da u è proporzionale a:
                d(u,v)^(-alpha)
    """

    if seed is not None:
        np.random.seed(seed)

    net = GridNetwork(n)

    for i in range(n):
        for j in range(n):

            src = (i, j)
            candidates = []
            weights = []

            for r in range(n):
                for c in range(n):

                    dst = (r, c)

                    if dst == src:
                        continue

                    d = GridNetwork.manhattan(src, dst)

                    candidates.append(dst)
                    weights.append(d ** (-alpha))

            weights = np.array(weights, dtype=float)
            weights /= weights.sum()

            for _ in range(num_links_per_node):

                index = np.random.choice(
                    len(candidates),
                    p=weights
                )

                dst = candidates[index]

                net.add_long_link(src, dst)

    return net