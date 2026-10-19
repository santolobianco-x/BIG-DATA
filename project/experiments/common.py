from src.evaluation import hits_at_k

def evaluate_baseline_hits(
        G,
        query_positive_edges,
        query_negative_edges,
        scoring_function,
        name,
        k_values=(10, 50, 100),
):
    """
    Calcola Hits@K di un baseline su un insieme di query.

    Parameters
    ----------
    G : networkx.Graph
        Grafo utilizzato per calcolare gli score.
        Per la valutazione deve essere il grafo di training.

    query_positive_edges : iterable
        Un arco positivo per ogni query.

    query_negative_edges : iterable of iterables
        Archi negativi candidati per ciascuna query.

    scoring_function : callable
        Funzione con firma scoring_function(G, edges).

    name : str
        Nome del baseline, utilizzato nel logging.

    k_values : iterable of int
        Valori di K per i quali calcolare Hits@K.

    Returns
    -------
    dict
        Dizionario {K: Hits@K}.
    """
    # Initial logic and parameters are validated
    query_positive_edges = list(query_positive_edges)
    query_negative_edges = [
        list(edges)
        for edges in query_negative_edges
    ]
    k_values = tuple(k_values)

    if not query_positive_edges:
        raise ValueError("L'insieme delle query positive è vuoto.")

    if len(query_positive_edges) != len(query_negative_edges):
        raise ValueError(
            "Il numero di query positive deve coincidere "
            "con quello degli insiemi di candidati negativi."
        )

    if not k_values:
        raise ValueError("Specificare almeno un valore di K.")

    if any(k <= 0 for k in k_values):
        raise ValueError("Tutti i valori di K devono essere positivi.")

    print()
    print(f"Computing Hits@K for {name}...")

    positive_scores = scoring_function(
        G,
        query_positive_edges,
    )

    if len(positive_scores) != len(query_positive_edges):
        raise ValueError(
            f"{name}: numero di score positivi non coerente "
            "con il numero di query."
        )

    # Vettorizzazione: valutazione flat e ricostruzione
    flat_negative_edges = [
        edge
        for query_edges in query_negative_edges
        for edge in query_edges
    ]

    flat_scores = scoring_function(
        G,
        flat_negative_edges,
    )

    if len(flat_scores) != len(flat_negative_edges):
        raise ValueError(
            f"{name}: numero di score negativi restituiti non coerente."
        )

    n_negatives = len(query_negative_edges[0])
    negative_scores = [
        flat_scores[i * n_negatives : (i + 1) * n_negatives]
        for i in range(len(query_positive_edges))
    ]

    results = {}
    for k in k_values:
        results[k] = hits_at_k(
            positive_scores,
            negative_scores,
            k,
        )

    return results