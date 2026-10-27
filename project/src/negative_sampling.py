"""
MODULO DEDICATO ALLA GENERAZIONE DEGLI ARCHI NEGATIVI

--> Per poter addestrare un modello a distinguere collegamenti veri da quelli
falsi, si devono generare artificialmente degli esempi negativi.
"""

import random


def generate_negative_edges(G, num_edges, seed=42, forbidden_edges=None):
    """
    Genera archi negativi casuali senza duplicati.

    Gli archi generati:
    - non devono appartenere a G;
    - non devono appartenere a forbidden_edges;
    - non devono essere duplicati.
    """

    if num_edges < 0:
        raise ValueError("num_edges non può essere negativo.")

    if num_edges == 0:
        return []

    nodes = list(G.nodes())

    if len(nodes) < 2:
        raise ValueError(
            "Servono almeno due nodi per generare archi negativi."
        )

    if forbidden_edges is None:
        forbidden_edges = set()
    else:
        forbidden_edges = {
            tuple(sorted(edge))
            for edge in forbidden_edges
        }

    # numero massimo di coppie non orientate fra nodi distinti.
    max_possible = len(nodes) * (len(nodes) - 1) // 2
    
    # coppie vietate + archi veri
    unavailable = {
        tuple(sorted((u, v)))
        for u, v in G.edges()
        if u != v
    }
    unavailable.update(
        edge for edge in forbidden_edges if edge[0] != edge[1]
    )

    available = max_possible - len(unavailable)

    if num_edges > available:
        raise ValueError(
            f"Richiesti {num_edges} archi negativi, "
            f"ma le coppie disponibili sono solo {available}."
        )

    rng = random.Random(seed)
    negative_edges = set()

    while len(negative_edges) < num_edges:
        u, v = rng.sample(nodes, 2)
        edge = tuple(sorted((u, v)))

        if edge in unavailable or edge in negative_edges:
            continue

        negative_edges.add(edge)

    return list(negative_edges) # lista di archi negativi


def generate_query_negative_edges(G, positive_edges, num_negatives_per_query=999, seed=42):
    """
    Genera candidati negativi distinti per ogni query positiva.
    Le coppie generate non devono essere archi del grafo.
    """

    if num_negatives_per_query < 0:
        raise ValueError(
            "num_negatives_per_query non può essere negativo."
        )

    rng = random.Random(seed)
    nodes = list(G.nodes())
    query_negative_edges = []

    for u, v in positive_edges:
        if u not in G:
            raise ValueError(f"Il nodo {u} non appartiene al grafo.")

        # Destinazioni ammissibili: nodi diversi da u
        # che non sono già suoi vicini.
        available = (
            len(nodes)
            - 1
            - len(set(G.neighbors(u)) - {u})
        )

        if num_negatives_per_query > available:
            raise ValueError(
                f"Query ({u}, {v}): richiesti "
                f"{num_negatives_per_query} negativi, "
                f"ma per il nodo {u} ne sono disponibili "
                f"solo {available}."
            )

        negative_edges = set()

        while len(negative_edges) < num_negatives_per_query:
            target = rng.choice(nodes)

            if target == u:
                continue

            if G.has_edge(u, target):
                continue

            edge = tuple(sorted((u, target)))
            negative_edges.add(edge)

        query_negative_edges.append(list(negative_edges))

    return query_negative_edges
