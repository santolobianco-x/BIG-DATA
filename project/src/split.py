import networkx as nx
import random


def split_edges(G: nx.Graph, test_ratio = 0.1, val_ratio = 0.1, seed =42):
    edges = list(G.edges())


    random.seed(seed)
    random.shuffle(edges)


    n_edges = len(edges)

    n_test = int(n_edges * test_ratio)
    n_val = int(n_edges * val_ratio)

    test_edges = edges[:n_test]
    val_edges = edges[n_test: n_test + n_val]
    train_edges = edges[n_test + n_val:]

    train_graph = nx.Graph()
    train_graph.add_nodes_from(G.nodes())
    train_graph.add_edges_from(train_edges)

    return train_graph, val_edges, test_edges