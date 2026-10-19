import networkx as nx
import random


def split_edges(G: nx.Graph, test_ratio = 0.1, val_ratio = 0.1, seed =42):
    edges = list(G.edges())


    rng = random.Random(seed)
    rng.shuffle(edges)


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


def split_train_edges_for_gnn(
    train_graph: nx.Graph,
    supervision_ratio=0.1,
    seed=42
):
    """
    Separa gli archi del training in:
    - message_passing_edges: usati per costruire il grafo osservabile;
    - supervised_train_edges: usati come positivi nella loss del decoder.

    Il grafo restituito mantiene tutti i nodi originali.
    """

    if not 0 < supervision_ratio < 1:
        raise ValueError(
            "supervision_ratio deve essere compreso tra 0 e 1."
        )

    edges = list(train_graph.edges())

    if len(edges) < 2:
        raise ValueError(
            "Servono almeno due archi per effettuare la separazione."
        )

    rng = random.Random(seed)
    rng.shuffle(edges)

    n_supervised = max(
        1,
        int(len(edges) * supervision_ratio)
    )

    supervised_train_edges = edges[:n_supervised]
    message_passing_edges = edges[n_supervised:]

    # Manteniamo tutti i nodi, anche quelli senza archi.
    message_passing_graph = nx.Graph()
    message_passing_graph.add_nodes_from(train_graph.nodes())
    message_passing_graph.add_edges_from(message_passing_edges)

    return (
        message_passing_graph,
        message_passing_edges,
        supervised_train_edges
    )
