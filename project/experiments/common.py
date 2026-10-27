"""
MODULO CONDIVISO PER LA VALUTAZIONE DELLE BASELINE TRADIZIONALI

--> valuta in modo comune per le baseline la metrica hits@K
"""

from src.evaluation import hits_at_k

def evaluate_baseline_hits(G, query_positive_edges,query_negative_edges,
        scoring_function, name, k_values=(10, 50, 100)):
    """
    Calcola la metrica Hits@K per una specifica baseline tradizionale.
    Vengono valutate con lo stesso identico rigore, lo stesso meccanismo
    di appiattimento dei distrattori e le stesse penalità sui pareggi.

    - scoring_function --> funzione euristica da valutare
    """

    # standardizzazione dei tipi per evitare bug
    query_positive_edges = list(query_positive_edges)


    query_negative_edges = [
        list(edges)
        for edges in query_negative_edges
    ]

    k_values = tuple(k_values)


    # controllo eventuali errori
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


    # applica la funzione di scoring
    positive_scores = scoring_function(G, query_positive_edges,)

    if len(positive_scores) != len(query_positive_edges):
        raise ValueError(
            f"{name}: numero di score positivi non coerente "
            "con il numero di query."
        )

    # si utilizza un'unica lista per sfruttare la vettorizzazione interna
    flat_negative_edges = [
        edge
        for query_edges in query_negative_edges
        for edge in query_edges
    ]

    # calcolo dello scoring per i negativi
    flat_scores = scoring_function(G, flat_negative_edges)

    if len(flat_scores) != len(flat_negative_edges):
        raise ValueError(
            f"{name}: numero di score negativi restituiti non coerente."
        )



    # si ricostruisce la lista tagliandola a fette 
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