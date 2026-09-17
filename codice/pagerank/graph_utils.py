import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def load_graph(path:str = "data/unict.txt"):
    G = nx.read_edgelist(path)
    G = nx.DiGraph(G)
    return G



def print_graph_info(G: nx.DiGraph, directed=True):
    print("Numero di nodi: ", len(G.nodes))
    print("Numero di archi: ", len(G.edges))

    if directed:
        print(
            "in-degree medio: ",
            sum(dict(G.in_degree).values()) / len(G.nodes)
        )
        print(
            "out-degree medio: ",
            sum(dict(G.out_degree).values()) / len(G.nodes)
        )
    else:
        print(
            "degree medio: ",
            sum(dict(G.degree).values()) / len(G.nodes)
        )



def power_law(x, a, b):
    return a * np.power(x, -b)


def get_in_degree(G:nx.DiGraph):
    return [d for n, d in G.in_degree()]

def get_out_degree(G:nx.DiGraph):
    return [d for n, d in G.out_degree()]


def plot_and_fit(degrees, title):
    values, counts = np.unique(degrees, return_counts=True)

    mask = values > 0

    x = values[mask]
    y = counts[mask]

    try:
        popt, _ = curve_fit(
            power_law,
            x,
            y,
            maxfev=10000
        )

        y_fit = power_law(x, *popt)

        fit_label = f"Fit: a={popt[0]:.2f}, b={popt[1]:.2f}"

    except RuntimeError:
        y_fit = np.zeros_like(x)
        fit_label = "Fit fallito"

    plt.figure()

    plt.loglog(x, y, 'bo', label="Dati")

    if fit_label != "Fit fallito":
        plt.loglog(x, y_fit, 'r-', label=fit_label)

    plt.xlabel("Degree")
    plt.ylabel("Frequenza")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def get_largest_component(G: nx.DiGraph):
    largest_component = max(nx.weakly_connected_components(G), key = len)
    return G.subgraph(largest_component).copy()

def get_pagerank(G: nx.DiGraph):
    return nx.pagerank(G)


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )