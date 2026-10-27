"""
MODULO DI CONVERSIONE E PREPARAZIONE DATI PER PYTORCH GEOMETRIC

--> funge da ponte tra la rappresentazione classica dei grafi(NetworkX) e
quella tensoriale ottimizzata per il Deep Learning(PyG).
"""

import random
import networkx as nx
import torch
from torch_geometric.data import Data

def prepare_edges(positive_edges, negative_edges, node_to_idx):
    """
    Crea i due tensori 'edge_index' e 'labels'(contenente le
    etichette target 1 per archi veri e 0 per archi non esistenti)

    node_to_idx è una mappatura degli ID originali con indici continui 0...N-1

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

    # CASO LIMITE --> ASSENZA DI ARCHI
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

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # .t() --> traspone la matrice
    # .contiguos() --> alloca i dati vicini in RAM

    labels = torch.tensor(labels, dtype=torch.float)

    return edge_index, labels


def sample_message_passing_edges(edge_index, keep_ratio, seed):
    """
    Sottocampiona casualmente gli archi per il message passing. Rimuovendo
    casualmente un arco ad ogni epoca, si costringe la GNN a non affidarsi 
    troppo a connessioni specifiche, rendendola più robusta.

    keep_ratio è la percentuale di archi da mantenere.
    """


    # CONTROLLI UTILI AD EVITARE OUT-BOUNDING E A RIDURRE IL LAVORO SE INUTILE
    if not 0.0 <= keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be between 0 and 1.")

    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)

    if keep_ratio == 1.0:
        return edge_index

    if keep_ratio == 0.0:
        return torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)

    edges = edge_index.t().tolist()


    # SI RICONDUCE TUTTO AD ARCHI NON ORIENTATI PER EVITARE DI DROPPARE 
    # UNA DIREZIONE SI E UNA NO
    undirected_edges = set()

    for u, v in edges:
        edge = tuple(sorted((u, v)))
        undirected_edges.add(edge)

    undirected_edges = list(undirected_edges)

    rng = random.Random(seed)

    rng.shuffle(undirected_edges)

    n_keep = int(len(undirected_edges) * keep_ratio)

    selected_edges = (undirected_edges[:n_keep])

    directed_edges = []

    for u, v in selected_edges:
        directed_edges.append([u, v])
        directed_edges.append([v, u])

    if len(directed_edges) == 0:
        return torch.empty((2, 0), dtype=edge_index.dtype,device=edge_index.device)

    return torch.tensor(directed_edges, dtype=edge_index.dtype,device=edge_index.device).t().contiguous()



def networkx_to_pyg(G, feature_mode="full"):
    """
    Converte un grafo NetworkX in un grafo PyTorch Geometric.

    feature_mode :
        modalità delle feature:
        - "full": degree + clustering
        - "degree": solo degree
        - "clustering": solo clustering
        - "constant": feature costante
    """

    if feature_mode not in {
        "full",
        "degree",
        "clustering",
        "constant"
    }:
        raise ValueError(
            f"Invalid feature mode: {feature_mode}"
        )

    nodes = list(G.nodes())

    node_to_idx = {node: i for i, node in enumerate(nodes)}

    edges = []

    for u, v in G.edges():
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        # In PyG, i grafi non orientati si simulano inserendo 
        # esplicitamente due archi orientati speculari (A -> B e B -> A)
        edges.append([u_idx, v_idx])
        edges.append([v_idx, u_idx])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    if feature_mode in {"full", "degree"}:
        degree = dict(G.degree())
        max_degree = max(degree.values(), default=0)
        # si calcola il grado normalizzato attraverso max(per evitare l'esplosione dei gradienti)
        if max_degree > 0:
            normalized_degree = {node: degree[node] / max_degree for node in nodes}
        else:
            normalized_degree = {node: 0.0 for node in nodes}

    # Calcolo coefficiente di clustering
    if feature_mode in {"full", "clustering"}:
        clustering = nx.clustering(G)

    # COSTRUZIONE MATRICE DI FEATURE 
    if feature_mode == "full":
        features = [[normalized_degree[node], clustering[node]]
                    for node in nodes]

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
        features = [[1.0] for node in nodes]

    x = torch.tensor(features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    """
    Ritorna:
    - data: l'oggetto standard PyG contenente 'x' ed 'edge_index'
    - node_to_idx: dizionario per mappare i nodi nel formato PyG
    """
    return data, node_to_idx 
