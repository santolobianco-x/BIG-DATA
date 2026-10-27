"""
MODULO PREPOSTO ALL'ANALISI DEL DEGREE BIAS.

--> indaga se i modelli favoriscono gli hubs a discapito dei nodi 
    a basso grado. Suddivide gli archi in 4 quartili, basati sul grado medio,
    e calcola AUC e AP separatamente.
"""
import math
import networkx as nx

from sklearn.metrics import (
roc_auc_score,
average_precision_score,
)

from src.evaluation import to_list


# ETICHETTE GLOBALI PER LE 4 FASCE
QUARTILES = ("Q1", "Q2", "Q3", "Q4")



def compute_edge_degree(G: nx.Graph, edges):
    """
    Calcola la popolarità di un arco basata sul grado medio dei sui estremi
    Il grado viene calcolato sul grafo di training per evitare Leakage.
    """

    edge_degrees = []

    for u, v in edges:
        degree_u = G.degree(u)
        degree_v = G.degree(v)

        # la popolarità dell'arco è la media dei gradi dei due nodi
        edge_degrees.append((degree_u + degree_v) / 2)

    return edge_degrees




def assign_degree_quartiles(edge_degrees):
    """
    Assegna ogni arco a una fascia basata sulla popolarità.
    - Q1 --> bassa popolarità
    - Q4 --> alta popolarità
    """

    edge_degrees = list(edge_degrees)

    if not edge_degrees:
        return []
    
    # Ordina i gradi per calcolare i percentili
    sorted_degrees = sorted(edge_degrees)
    n = len(sorted_degrees)

    q1 = sorted_degrees[int(n * 0.25)]
    q2 = sorted_degrees[int(n * 0.50)]
    q3 = sorted_degrees[int(n * 0.75)]

    quartiles = []

    # Assegnazione di ogni arco alla propria fascia
    for degree in edge_degrees:
        if degree <= q1:
            quartiles.append("Q1") # nodi periferici
        elif degree <= q2:
            quartiles.append("Q2")
        elif degree <= q3:
            quartiles.append("Q3")
        else:
            quartiles.append("Q4") # nodi hub 

    return quartiles # lista di stringhe("Q1", "Q2", "Q3", "Q4") allineate con l'input


def evaluate_degree_bias(positive_edges, negative_edges, positive_scores,
                         negative_scores, G: nx.Graph):

    """
    Slicing delle metriche di valutazione per fascia di popolrità.

    Unisce gli archi veri e falsi per calocolare soglie di quartile eque,
    dopodiché separa le predizioni del modello in 4 sottomultipli e calcola
    AUC/AP per ognuno.
    """

    # standardizzazione dei dati 
    positive_edges = list(positive_edges)
    negative_edges = list(negative_edges)

    positive_scores = to_list(positive_scores)
    negative_scores = to_list(negative_scores)

    if len(positive_edges) != len(positive_scores):
        raise ValueError("Number of positive edges and positive scores is inconsistent.")

    if len(negative_edges) != len(negative_scores):
        raise ValueError("Number of negative edges and negative scores is inconsistent.")

    # si uniscono gli archi per avere una distribuzione globale dei gradi
    all_edges = positive_edges + negative_edges

    # calcola la popolarità
    all_degrees = compute_edge_degree(G, all_edges)
    quartiles = assign_degree_quartiles(all_degrees)

    # si divide per score positivi e negativi
    positive_count = len(positive_edges)
    positive_quartiles = quartiles[:positive_count]
    negative_quartiles = quartiles[positive_count:]

    results = {}

    # si calcolano le metriche separatamente per ogni fascia
    for quartile in QUARTILES:
        # si cercano gli indici degli archi positivi che appartengono a questo quartile
        positive_indices = [
            i for i, group in enumerate(positive_quartiles) if group == quartile
        ]

        # si cercano gli indici degli archi negativi che appartengono a questo quartile
        negative_indices = [
            i for i, group in enumerate(negative_quartiles) if group == quartile
        ]

        # si prendono gli score che sono legati agli indici degli archi
        q_positive_scores = [positive_scores[i] for i in positive_indices]
        q_negative_scores = [negative_scores[i] for i in negative_indices]

        # se in un quartile mancano i positivi o i negativi
        # restituisce NaN
        if (not q_positive_scores or not q_negative_scores):
            results[quartile] = {
                "positive_edges": len(q_positive_scores),
                "negative_edges": len(q_negative_scores),
                "auc": math.nan,
                "ap": math.nan,
            }
            continue

        # concatenazione degli score
        scores = q_positive_scores + q_negative_scores

        # concatenazione delle etichette
        labels = [1] * len(q_positive_scores) + [0] * len(q_negative_scores)

        results[quartile] = {
            "positive_edges": len(q_positive_scores),
            "negative_edges": len(q_negative_scores),
            "auc": roc_auc_score(labels, scores),
            "ap": average_precision_score(labels, scores),
        }

    return results


def run_degree_bias_experiment(train_graph: nx.Graph, positive_edges,
        negative_edges, scoring_functions: dict):
    """
    Orchestra l'esperimento di Degree Bias per tutte le basiline
    configurate e stampa una tabella riassuntiva dei risultati.
    """

    positive_edges = list(positive_edges)
    negative_edges = list(negative_edges)
 
    if not positive_edges or not negative_edges:
        raise ValueError( "The test set must contain both positive and negative edges.")

    if not scoring_functions:
        raise ValueError("Specify at least one baseline to evaluate.")

    print("\n" + "=" * 65)
    print("DEGREE-BIAS ANALYSIS")
    print("=" * 65)

    print("Endpoint degrees are computed exclusively on the training graph.")

    all_results = {}

    # itera sui modelli e calcola il bias
    for name, scoring_function in scoring_functions.items():
        print(f"\nBaseline: {name}")

        positive_scores = scoring_function(train_graph, positive_edges)

        negative_scores = scoring_function(train_graph, negative_edges)

        results = evaluate_degree_bias(
            positive_edges=positive_edges,
            negative_edges=negative_edges,
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            G=train_graph,
        )

        all_results[name] = results

        # stampa l'intestazione della tabella
        print(f"{'Quartile':<10}"
            f"{'Positives':>12}"
            f"{'Negatives':>12}"
            f"{'AUC':>12}"
            f"{'AP':>12}"
        )

        # stampa le righe formattate(numeri o NaN)
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

    return all_results # dizionario contenente i risultati stratificati 

