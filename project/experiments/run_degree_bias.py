import math
import networkx as nx

from sklearn.metrics import (
roc_auc_score,
average_precision_score,
)

from src.evaluation import to_list

QUARTILES = ("Q1", "Q2", "Q3", "Q4")



def compute_edge_degree(G: nx.Graph, edges):
    """
    Calcola il grado medio degli estremi di ogni arco.

    ```
    Il grado è calcolato sul grafo G, che nell'esperimento
    deve essere il grafo di training.
    """

    edge_degrees = []

    for u, v in edges:
        degree_u = G.degree(u)
        degree_v = G.degree(v)

        edge_degrees.append(
            (degree_u + degree_v) / 2
        )

    return edge_degrees


def assign_degree_quartiles(edge_degrees):
    """
    Assegna gli archi a Q1, Q2, Q3 o Q4 in base al grado medio.

    ```
    I valori uguali ricevono la stessa assegnazione.
    Per questo motivo, in presenza di molti pareggi,
    i quartili possono avere dimensioni diverse o essere vuoti.
    """

    edge_degrees = list(edge_degrees)

    if not edge_degrees:
        return []

    sorted_degrees = sorted(edge_degrees)
    n = len(sorted_degrees)

    q1 = sorted_degrees[int(n * 0.25)]
    q2 = sorted_degrees[int(n * 0.50)]
    q3 = sorted_degrees[int(n * 0.75)]

    quartiles = []

    for degree in edge_degrees:
        if degree <= q1:
            quartiles.append("Q1")
        elif degree <= q2:
            quartiles.append("Q2")
        elif degree <= q3:
            quartiles.append("Q3")
        else:
            quartiles.append("Q4")

    return quartiles


def evaluate_degree_bias(
        positive_edges,
        negative_edges,
        positive_scores,
        negative_scores,
        G: nx.Graph,):
    """
    Valuta AUC e AP per quartile di grado medio degli estremi.

    ```
    Le soglie dei quartili sono determinate considerando
    insieme gli archi positivi e negativi.

    G deve essere il grafo di training.
    """

    positive_edges = list(positive_edges)
    negative_edges = list(negative_edges)

    positive_scores = to_list(positive_scores)
    negative_scores = to_list(negative_scores)

    if len(positive_edges) != len(positive_scores):
        raise ValueError(
            "Numero di archi positivi e score positivi non coerente."
        )

    if len(negative_edges) != len(negative_scores):
        raise ValueError(
            "Numero di archi negativi e score negativi non coerente."
        )

    all_edges = positive_edges + negative_edges

    all_degrees = compute_edge_degree(
        G,
        all_edges,
    )

    quartiles = assign_degree_quartiles(all_degrees)

    positive_count = len(positive_edges)

    positive_quartiles = quartiles[:positive_count]
    negative_quartiles = quartiles[positive_count:]

    results = {}

    for quartile in QUARTILES:
        positive_indices = [
            i
            for i, group in enumerate(positive_quartiles)
            if group == quartile
        ]

        negative_indices = [
            i
            for i, group in enumerate(negative_quartiles)
            if group == quartile
        ]

        q_positive_scores = [
            positive_scores[i]
            for i in positive_indices
        ]

        q_negative_scores = [
            negative_scores[i]
            for i in negative_indices
        ]

        if (
            not q_positive_scores
            or not q_negative_scores
        ):
            results[quartile] = {
                "positive_edges": len(q_positive_scores),
                "negative_edges": len(q_negative_scores),
                "auc": math.nan,
                "ap": math.nan,
            }
            continue

        scores = q_positive_scores + q_negative_scores

        labels = (
            [1] * len(q_positive_scores)
            + [0] * len(q_negative_scores)
        )

        results[quartile] = {
            "positive_edges": len(q_positive_scores),
            "negative_edges": len(q_negative_scores),
            "auc": roc_auc_score(labels, scores),
            "ap": average_precision_score(labels, scores),
        }

    return results


def run_degree_bias_experiment(
        train_graph: nx.Graph,
        positive_edges,
        negative_edges,
        scoring_functions: dict,
        ):
    """
    Esegue l'analisi degree-bias sui baseline specificati.

    ```
    Parameters
    ----------
    train_graph : networkx.Graph
        Grafo di training, usato sia per gli score
        sia per calcolare i gradi degli estremi.

    positive_edges, negative_edges : iterable
        Archi positivi e negativi del test.

    scoring_functions : dict
        Dizionario {nome_baseline: funzione_di_scoring}.

    Returns
    -------
    dict
        Risultati per baseline e quartile.
    """

    positive_edges = list(positive_edges)
    negative_edges = list(negative_edges)

    if not positive_edges or not negative_edges:
        raise ValueError(
            "Il test deve contenere archi positivi e negativi."
        )

    if not scoring_functions:
        raise ValueError(
            "Specificare almeno un baseline da valutare."
        )

    print("\n" + "=" * 65)
    print("DEGREE-BIAS ANALYSIS")
    print("=" * 65)

    print(
        "Il grado degli estremi viene calcolato "
        "esclusivamente sul grafo di training."
    )

    all_results = {}

    for name, scoring_function in scoring_functions.items():
        print(f"\nBaseline: {name}")

        positive_scores = scoring_function(
            train_graph,
            positive_edges,
        )

        negative_scores = scoring_function(
            train_graph,
            negative_edges,
        )

        results = evaluate_degree_bias(
            positive_edges=positive_edges,
            negative_edges=negative_edges,
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            G=train_graph,
        )

        all_results[name] = results

        print(
            f"{'Quartile':<10}"
            f"{'Positivi':>12}"
            f"{'Negativi':>12}"
            f"{'AUC':>12}"
            f"{'AP':>12}"
        )

        for quartile in QUARTILES:
            row = results[quartile]

            auc_text = (
                f"{row['auc']:.4f}"
                if math.isfinite(row["auc"])
                else "N/A"
            )

            ap_text = (
                f"{row['ap']:.4f}"
                if math.isfinite(row["ap"])
                else "N/A"
            )

            print(
                f"{quartile:<10}"
                f"{row['positive_edges']:>12}"
                f"{row['negative_edges']:>12}"
                f"{auc_text:>12}"
                f"{ap_text:>12}"
            )

    print("\nDegree-bias analysis completed.")

    return all_results

