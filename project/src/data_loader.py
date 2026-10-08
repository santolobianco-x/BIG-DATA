import networkx as nx


def load_graph(path):
    """
    Carica il dataset come grafo non orientato
    """
    G = nx.read_edgelist(
        path,
        nodetype=int,
        comments='#'
    )

    return G