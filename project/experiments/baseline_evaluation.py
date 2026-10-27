"""
MODULO PER L'ORCHESTRAZIONE DELLA VALUTAZIONE DELLE BASELINE

--> automatizza l'intero processo di test dei metodi tradizionali.
    Itera sul dizionario delle euristiche, calocla gli score sul grafo
    e compila un report completo delle metriche globali e di ranking.
"""
from config import HITS_K_VALUES
from src.evaluation import evaluate_scores
from experiments.common import evaluate_baseline_hits


def evaluate_baselines(graph, test_edges, test_negative_edges,
    query_positive_edges, query_negative_edges,baseline_methods,
    title,print_section):

    """
    Esegue la pipeline completa di valutazione per tutti i 
    metodi classici forniti.
    """

    print_section(title)

    results = {}
    hits_results = {}

    # itera dinamicamente su tutte le funzioni matematiche configurate
    for name, scoring_function in baseline_methods.items():
        print()
        print("-" * 70)
        print(name)
        print("-" * 70)

        # scoring di connessione per gli archi veri e falsi
        positive_scores = scoring_function(graph, test_edges)
        negative_scores = scoring_function(graph, test_negative_edges)

        # valuta gli archi attraverso le metriche AUC e AP
        metrics = evaluate_scores(positive_scores, negative_scores)

        results[name] = {
            "auc": metrics["auc"],
            "ap": metrics["ap"],
        }

        print(f"Test AUC: {metrics['auc']:.6f}")
        print(f"Test AP:  {metrics['ap']:.6f}")

        # calcola Hits@K per la baseline presa in considerazione
        hits = evaluate_baseline_hits(
            graph,
            query_positive_edges,
            query_negative_edges,
            scoring_function,
            name,
            k_values=HITS_K_VALUES,
        )

        hits_results[name] = hits

        for k in HITS_K_VALUES:
            print(f"Hits@{k}: {hits[k]:.4f}")

    
    return results, hits_results # restituisce i dizionari pronti per essere conservati

