import networkx as nx
import torch
from torch_geometric.data import Data


def networkx_to_pyg(G, feature_mode="full"):
    """
    Converte un grafo NetworkX in un grafo PyTorch Geometric.

    Parameters
    ----------
    G : networkx.Graph
        Grafo da convertire.

    feature_mode : str
        Modalità delle feature:
        - "full": degree + clustering
        - "degree": solo degree
        - "clustering": solo clustering
        - "constant": feature costante

    Returns
    -------
    data : torch_geometric.data.Data
        Grafo PyG con feature nodali ed edge_index.

    node_to_idx : dict
        Mappa node ID -> indice PyTorch Geometric.
    """

    if feature_mode not in {
        "full",
        "degree",
        "clustering",
        "constant"
    }:
        raise ValueError(
            f"Feature mode non valido: {feature_mode}"
        )

    # --------------------------------------------------
    # NODE MAPPING
    # --------------------------------------------------

    nodes = list(G.nodes())

    node_to_idx = {
        node: i
        for i, node in enumerate(nodes)
    }

    # --------------------------------------------------
    # EDGE INDEX
    # --------------------------------------------------

    edges = []

    for u, v in G.edges():

        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]

        # Grafo non orientato:
        # inseriamo entrambe le direzioni.
        edges.append([u_idx, v_idx])
        edges.append([v_idx, u_idx])

    edge_index = torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()

    # --------------------------------------------------
    # NODE FEATURES
    # --------------------------------------------------

    degree = dict(G.degree())
    clustering = nx.clustering(G)

    max_degree = max(degree.values())

    if max_degree > 0:
        normalized_degree = {
            node: degree[node] / max_degree
            for node in nodes
        }
    else:
        normalized_degree = {
            node: 0.0
            for node in nodes
        }

    if feature_mode == "full":

        features = [
            [
                normalized_degree[node],
                clustering[node]
            ]
            for node in nodes
        ]

    elif feature_mode == "degree":

        features = [
            [normalized_degree[node]]
            for node in nodes
        ]

    elif feature_mode == "clustering":

        features = [
            [clustering[node]]
            for node in nodes
        ]

    elif feature_mode == "constant":

        features = [
            [1.0]
            for node in nodes
        ]

    x = torch.tensor(
        features,
        dtype=torch.float
    )

    data = Data(
        x=x,
        edge_index=edge_index
    )

    return data, node_to_idx