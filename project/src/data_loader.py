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

    
    G.remove_edges_from(nx.selfloop_edges(G))

    return G