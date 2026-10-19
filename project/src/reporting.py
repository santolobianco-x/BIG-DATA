def print_section(title):
    """Stampa un titolo di sezione."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_graph_summary(graph, graph_statistics, degree_statistics):
    """Stampa le statistiche generali e di grado del grafo."""
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
    """Stampa AUC, AP e Hits@K delle baseline."""
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
