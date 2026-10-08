import random
import networkx as nx
import torch
from torch_geometric.data import Data

def prepare_edges(
    positive_edges,
    negative_edges,
    node_to_idx
):
    """
    Converte archi NetworkX in edge_index PyTorch Geometric.

    Gli archi positivi vengono assegnati label 1.
    Gli archi negativi vengono assegnati label 0.
    """

    edges = []
    labels = []

    for u, v in positive_edges:
        edges.append([
            node_to_idx[u],
            node_to_idx[v]
        ])
        labels.append(1.0)

    for u, v in negative_edges:
        edges.append([
            node_to_idx[u],
            node_to_idx[v]
        ])
        labels.append(0.0)

    if len(edges) == 0:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

        labels = torch.empty(
            (0,),
            dtype=torch.float
        )

        return edge_index, labels

    edge_index = torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()

    labels = torch.tensor(
        labels,
        dtype=torch.float
    )

    return edge_index, labels


def sample_message_passing_edges(
    edge_index,
    keep_ratio,
    seed
):
    """
    Seleziona casualmente una percentuale degli archi
    utilizzati per il message passing.
    """

    if not 0.0 <= keep_ratio <= 1.0:
        raise ValueError(
            "keep_ratio deve essere compreso tra 0 e 1."
        )

    if edge_index.numel() == 0:

        return torch.empty(
            (2, 0),
            dtype=edge_index.dtype,
            device=edge_index.device
        )

    if keep_ratio == 1.0:
        return edge_index

    if keep_ratio == 0.0:

        return torch.empty(
            (2, 0),
            dtype=edge_index.dtype,
            device=edge_index.device
        )

    edges = edge_index.t().tolist()

    undirected_edges = set()

    for u, v in edges:

        edge = tuple(sorted((u, v)))

        undirected_edges.add(edge)

    undirected_edges = list(
        undirected_edges
    )

    rng = random.Random(seed)

    rng.shuffle(
        undirected_edges
    )

    n_keep = int(
        len(undirected_edges)
        * keep_ratio
    )

    selected_edges = (
        undirected_edges[:n_keep]
    )

    directed_edges = []

    for u, v in selected_edges:

        directed_edges.append(
            [u, v]
        )

        directed_edges.append(
            [v, u]
        )

    if len(directed_edges) == 0:

        return torch.empty(
            (2, 0),
            dtype=edge_index.dtype,
            device=edge_index.device
        )

    return torch.tensor(
        directed_edges,
        dtype=edge_index.dtype,
        device=edge_index.device
    ).t().contiguous()



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