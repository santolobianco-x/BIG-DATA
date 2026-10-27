"""
MODULO DEDICATO ALLA FORMATTAZIONE E STAMPA DEI RISULTATI
"""

def print_section(title):
    print()
    print("=" * 70)
    print(title) # stampa un titolo di sezione
    print("=" * 70)



def print_graph_summary(graph, graph_statistics, degree_statistics):
    """Stampa le statistiche generali e di grado del grafo.
    Vengono passate le funzioni per calcolare le statistiche come parametri,
    rendendo questa funzione indipendente dall'implementazione matematica."""
    graph_stats = graph_statistics(graph)
    degree_stats = degree_statistics(graph)

    print(f"Nodes:       {graph_stats['nodes']}")
    print(f"Edges:       {graph_stats['edges']}")
    print(f"Density:     {graph_stats['density']}")
    print(f"Components:  {graph_stats['connected_components']}")
    print(f"Min degree:  {degree_stats['min_degree']}")
    print(f"Max degree:  {degree_stats['max_degree']}")
    print(f"Avg degree:  {degree_stats['average_degree']}")
    print(f"Median:      {degree_stats['median_degree']}")



def print_baseline_results(title, results, hits_results):
    """Stampa AUC, AP e Hits@K per i metodi delle baseline.
    - results (dict): dizionario con le metriche
    - hits_results (dict): dizionario con le metriche Hits@k per ogni baseline"""
    print_section(title)

    print(
        f"{'Method':<25}"
        f"{'AUC':>12}"
        f"{'AP':>12}"
        f"{'Hits@10':>12}"
        f"{'Hits@50':>12}"
        f"{'Hits@100':>12}"
    )
    print("-" * 85)

    for name, result in results.items():
        hits = hits_results[name]

        print(
            f"{name:<25}"
            f"{result['auc']:>12.4f}"
            f"{result['ap']:>12.4f}"
            f"{hits[10]:>12.4f}"
            f"{hits[50]:>12.4f}"
            f"{hits[100]:>12.4f}"
        )
