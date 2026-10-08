import math

import networkx as nx
import numpy as np
import torch

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

from src.gnn import (
    networkx_to_pyg,
    GraphSAGE,
    MLPDecoder,
    train_gnn,
    predict_edges,
    predict_query_candidates,
    prepare_edges
)


from src.evaluation import hits_at_k


# ============================================================
# GNN MULTI-SEED EXPERIMENT
# ============================================================

def run_gnn_multiseed(
    train_graph,
    train_edges,
    train_negative_edges,
    val_edges,
    val_negative_edges,
    test_edges,
    test_negative_edges,
    query_positive_edges,
    query_negative_edges,
    seeds=(1, 2, 3, 4, 5),
    hidden_channels=64,
    out_channels=32,
    decoder_hidden_dim=64,
    epochs=500,
    learning_rate=0.001
):
    """
    Esegue il modello GraphSAGE + MLP
    su più seed indipendenti.

    Le feature vengono costruite esclusivamente
    sul training graph per evitare data leakage.

    Per ogni seed vengono salvati:

        - best epoch
        - validation AUC
        - validation AP
        - test AUC
        - test AP
        - Hits@10
        - Hits@50
        - Hits@100

    Le metriche Hits@K vengono calcolate utilizzando
    esattamente le stesse query utilizzate per
    i baseline.

    Alla fine viene restituito un dizionario
    con i risultati di tutti i seed.
    """

    results = []

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    # Le feature vengono costruite SOLO dal training graph.
    data, node_to_idx = networkx_to_pyg(
        train_graph,
        feature_mode="full"
    )

    # --------------------------------------------------------
    # MULTI-SEED
    # --------------------------------------------------------

    for seed in seeds:

        print()
        print("=" * 70)
        print(f"GNN FULL - SEED {seed}")
        print("=" * 70)

        # ----------------------------------------------------
        # REPRODUCIBILITY
        # ----------------------------------------------------

        torch.manual_seed(seed)
        np.random.seed(seed)

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = GraphSAGE(
            in_channels=data.x.shape[1],
            hidden_channels=hidden_channels,
            out_channels=out_channels
        )

        decoder = MLPDecoder(
            embedding_dim=out_channels,
            hidden_dim=decoder_hidden_dim
        )

        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        optimizer = torch.optim.Adam(
            list(model.parameters())
            + list(decoder.parameters()),
            lr=learning_rate
        )

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        history = train_gnn(
            model=model,
            decoder=decoder,
            data=data,
            train_edges=train_edges,
            train_negative_edges=train_negative_edges,
            val_edges=val_edges,
            val_negative_edges=val_negative_edges,
            node_to_idx=node_to_idx,
            optimizer=optimizer,
            epochs=epochs,
            keep_ratio=1.0,
            seed=seed
        )

        # ----------------------------------------------------
        # TEST EDGE PREPARATION
        # ----------------------------------------------------

        test_pos_edge_index, test_pos_labels = prepare_edges(
            test_edges,
            [],
            node_to_idx
        )

        test_neg_edge_index, test_neg_labels = prepare_edges(
            [],
            test_negative_edges,
            node_to_idx
        )

        test_edge_index = torch.cat(
            [
                test_pos_edge_index,
                test_neg_edge_index
            ],
            dim=1
        )

        test_labels = torch.cat(
            [
                test_pos_labels,
                test_neg_labels
            ]
        )

        # ----------------------------------------------------
        # TEST PREDICTION
        # ----------------------------------------------------

        test_probabilities = predict_edges(
            model,
            decoder,
            data,
            test_edge_index
        )

        test_labels_numpy = (
            test_labels
            .detach()
            .cpu()
            .numpy()
        )

        test_probabilities_numpy = (
            test_probabilities
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # TEST METRICS
        # ----------------------------------------------------

        test_auc = roc_auc_score(
            test_labels_numpy,
            test_probabilities_numpy
        )

        test_ap = average_precision_score(
            test_labels_numpy,
            test_probabilities_numpy
        )

        # ----------------------------------------------------
        # QUERY-BASED HITS@K
        # ----------------------------------------------------

        print()
        print("Computing query-based Hits@K...")

        (
            query_positive_scores,
            query_negative_scores
        ) = predict_query_candidates(
            model=model,
            decoder=decoder,
            data=data,
            query_positive_edges=query_positive_edges,
            query_negative_edges=query_negative_edges,
            node_to_idx=node_to_idx
        )

        hits_10 = hits_at_k(
            query_positive_scores,
            query_negative_scores,
            10
        )

        hits_50 = hits_at_k(
            query_positive_scores,
            query_negative_scores,
            50
        )

        hits_100 = hits_at_k(
            query_positive_scores,
            query_negative_scores,
            100
        )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        result = {
            "seed": seed,
            "best_epoch": history["best_epoch"],
            "val_auc": history["best_val_auc"],
            "val_ap": history["best_val_ap"],
            "test_auc": test_auc,
            "test_ap": test_ap,
            "hits@10": hits_10,
            "hits@50": hits_50,
            "hits@100": hits_100
        }

        results.append(result)

        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print()
        print(f"Seed:       {seed}")
        print(f"Best epoch: {history['best_epoch']}")
        print(f"Val AUC:    {history['best_val_auc']:.4f}")
        print(f"Val AP:     {history['best_val_ap']:.4f}")
        print(f"Test AUC:   {test_auc:.4f}")
        print(f"Test AP:    {test_ap:.4f}")
        print(f"Hits@10:    {hits_10:.4f}")
        print(f"Hits@50:    {hits_50:.4f}")
        print(f"Hits@100:   {hits_100:.4f}")

    # ========================================================
    # MULTI-SEED SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("MULTI-SEED SUMMARY")
    print("=" * 70)

    val_auc_values = [
        result["val_auc"]
        for result in results
    ]

    val_ap_values = [
        result["val_ap"]
        for result in results
    ]

    test_auc_values = [
        result["test_auc"]
        for result in results
    ]

    test_ap_values = [
        result["test_ap"]
        for result in results
    ]

    hits_10_values = [
        result["hits@10"]
        for result in results
    ]

    hits_50_values = [
        result["hits@50"]
        for result in results
    ]

    hits_100_values = [
        result["hits@100"]
        for result in results
    ]

    # --------------------------------------------------------
    # MEAN / STD
    # --------------------------------------------------------

    val_auc_mean = np.mean(
        val_auc_values
    )

    val_auc_std = np.std(
        val_auc_values
    )

    val_ap_mean = np.mean(
        val_ap_values
    )

    val_ap_std = np.std(
        val_ap_values
    )

    test_auc_mean = np.mean(
        test_auc_values
    )

    test_auc_std = np.std(
        test_auc_values
    )

    test_ap_mean = np.mean(
        test_ap_values
    )

    test_ap_std = np.std(
        test_ap_values
    )

    hits_10_mean = np.mean(
        hits_10_values
    )

    hits_10_std = np.std(
        hits_10_values
    )

    hits_50_mean = np.mean(
        hits_50_values
    )

    hits_50_std = np.std(
        hits_50_values
    )

    hits_100_mean = np.mean(
        hits_100_values
    )

    hits_100_std = np.std(
        hits_100_values
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print(
        f"Val AUC  : "
        f"{val_auc_mean:.4f} ± {val_auc_std:.4f}"
    )

    print(
        f"Val AP   : "
        f"{val_ap_mean:.4f} ± {val_ap_std:.4f}"
    )

    print(
        f"Test AUC : "
        f"{test_auc_mean:.4f} ± {test_auc_std:.4f}"
    )

    print(
        f"Test AP  : "
        f"{test_ap_mean:.4f} ± {test_ap_std:.4f}"
    )

    print(
        f"Hits@10  : "
        f"{hits_10_mean:.4f} ± {hits_10_std:.4f}"
    )

    print(
        f"Hits@50  : "
        f"{hits_50_mean:.4f} ± {hits_50_std:.4f}"
    )

    print(
        f"Hits@100 : "
        f"{hits_100_mean:.4f} ± {hits_100_std:.4f}"
    )

    # --------------------------------------------------------
    # RANGE
    # --------------------------------------------------------

    print()
    print("Test AUC range:")
    print(
        f"  min = {np.min(test_auc_values):.4f}"
    )
    print(
        f"  max = {np.max(test_auc_values):.4f}"
    )

    print()
    print("Test AP range:")
    print(
        f"  min = {np.min(test_ap_values):.4f}"
    )
    print(
        f"  max = {np.max(test_ap_values):.4f}"
    )

    print()
    print("Hits@10 range:")
    print(
        f"  min = {np.min(hits_10_values):.4f}"
    )
    print(
        f"  max = {np.max(hits_10_values):.4f}"
    )

    print()
    print("Hits@50 range:")
    print(
        f"  min = {np.min(hits_50_values):.4f}"
    )
    print(
        f"  max = {np.max(hits_50_values):.4f}"
    )

    print()
    print("Hits@100 range:")
    print(
        f"  min = {np.min(hits_100_values):.4f}"
    )
    print(
        f"  max = {np.max(hits_100_values):.4f}"
    )

    return results


# ============================================================
# DEGREE BIAS ANALYSIS
# ============================================================

def compute_edge_degree(
    G: nx.Graph,
    edges
):
    """
    Calcola il grado medio dei due estremi di ogni arco.

    Per un arco (u, v):

        degree(u, v) = (deg(u) + deg(v)) / 2

    Il grado viene calcolato sul grafo G.
    """

    edge_degrees = []

    for u, v in edges:

        degree_u = G.degree(u)
        degree_v = G.degree(v)

        average_degree = (
            degree_u + degree_v
        ) / 2

        edge_degrees.append(
            average_degree
        )

    return edge_degrees


def assign_degree_quartiles(
    edge_degrees
):
    """
    Divide gli archi in quattro gruppi
    in base al grado medio dei loro estremi.

    Restituisce una lista contenente:

        Q1, Q2, Q3, Q4
    """

    if len(edge_degrees) == 0:
        return []

    sorted_degrees = sorted(
        edge_degrees
    )

    n = len(sorted_degrees)

    q1 = sorted_degrees[
        int(n * 0.25)
    ]

    q2 = sorted_degrees[
        int(n * 0.50)
    ]

    q3 = sorted_degrees[
        int(n * 0.75)
    ]

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
    G: nx.Graph
):
    """
    Analizza le prestazioni di un metodo di
    link prediction in funzione del grado
    dei nodi coinvolti negli archi.

    I quartili vengono definiti considerando
    insieme archi positivi e negativi.

    Restituisce AUC e AP per ciascun quartile.
    """

    all_edges = (
        list(positive_edges)
        + list(negative_edges)
    )

    all_degrees = compute_edge_degree(
        G,
        all_edges
    )

    quartiles = assign_degree_quartiles(
        all_degrees
    )

    positive_count = len(
        positive_edges
    )

    positive_quartiles = (
        quartiles[:positive_count]
    )

    negative_quartiles = (
        quartiles[positive_count:]
    )

    results = {}

    for quartile in [
        "Q1",
        "Q2",
        "Q3",
        "Q4"
    ]:

        positive_indices = [
            i
            for i, q in enumerate(
                positive_quartiles
            )
            if q == quartile
        ]

        negative_indices = [
            i
            for i, q in enumerate(
                negative_quartiles
            )
            if q == quartile
        ]

        q_positive_scores = [
            positive_scores[i]
            for i in positive_indices
        ]

        q_negative_scores = [
            negative_scores[i]
            for i in negative_indices
        ]

        # ----------------------------------------------------
        # MISSING CLASS
        # ----------------------------------------------------

        if (
            len(q_positive_scores) == 0
            or len(q_negative_scores) == 0
        ):

            results[quartile] = {
                "positive_edges": len(
                    q_positive_scores
                ),
                "negative_edges": len(
                    q_negative_scores
                ),
                "auc": math.nan,
                "ap": math.nan
            }

            continue

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        scores = (
            q_positive_scores
            + q_negative_scores
        )

        labels = (
            [1] * len(q_positive_scores)
            + [0] * len(q_negative_scores)
        )

        auc = roc_auc_score(
            labels,
            scores
        )

        ap = average_precision_score(
            labels,
            scores
        )

        results[quartile] = {
            "positive_edges": len(
                q_positive_scores
            ),
            "negative_edges": len(
                q_negative_scores
            ),
            "auc": auc,
            "ap": ap
        }

    return results