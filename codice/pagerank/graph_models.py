import networkx as nx

def generate_random_graph(n, p=0.00008, seed=1):
    return nx.fast_gnp_random_graph(
        n,
        p,
        seed=seed,
        directed=False
    )


def find_ba_m(n, target_edges):
    m = 1

    while m < n and m * (n - m) + m * (m - 1) // 2 < target_edges:
        m += 1

    return m


def generate_ba_graph(n, m, seed=1):
    return nx.barabasi_albert_graph(
        n,
        m,
        seed=seed
    )