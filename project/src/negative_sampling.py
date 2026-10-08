import networkx as nx
import random


def generate_negative_edges(
    G,
    num_edges,
    seed=42,
    forbidden_edges=None
):
    """
    Genera archi negativi casuali.

    Gli archi generati:
    - non devono appartenere a G;
    - non devono appartenere a forbidden_edges;
    - non devono essere duplicati.
    """

    rng = random.Random(seed)

    nodes = list(G.nodes())

    if forbidden_edges is None:
        forbidden_edges = set()
    else:
        forbidden_edges = {
            tuple(sorted(edge))
            for edge in forbidden_edges
        }

    negative_edges = set()

    while len(negative_edges) < num_edges:

        u, v = rng.sample(nodes, 2)

        edge = tuple(sorted((u, v)))

        if G.has_edge(u, v):
            continue

        if edge in forbidden_edges:
            continue

        if edge in negative_edges:
            continue

        negative_edges.add(edge)

    return list(negative_edges)


def generate_query_negative_edges(
    G,
    positive_edges,
    num_negatives_per_query=999,
    seed=42
):
    """
    Genera negativi associati a ciascun arco positivo.

    Per ogni query positiva (u, v), genera:

        (u, x1)
        (u, x2)
        ...
        (u, xN)

    dove gli archi negativi non esistono nel grafo originale.

    La generazione usa rejection sampling invece di
    costruire una lista di tutti i nodi disponibili per
    ogni query. Questo riduce notevolmente il consumo
    di memoria su grafi grandi.
    """

    if num_negatives_per_query <= 0:
        raise ValueError(
            "num_negatives_per_query deve essere maggiore di 0."
        )

    rng = random.Random(seed)

    nodes = list(G.nodes())

    query_negative_edges = []

    for u, v in positive_edges:

        negative_edges = set()

        while len(negative_edges) < num_negatives_per_query:

            target = rng.choice(nodes)

            if target == u:
                continue

            if G.has_edge(u, target):
                continue

            edge = tuple(sorted((u, target)))

            if edge in negative_edges:
                continue

            negative_edges.add(edge)

        query_negative_edges.append(
            list(negative_edges)
        )

    return query_negative_edges