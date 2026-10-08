import random

import numpy as np
import torch

from src.data_loader import load_graph

from src.graph_utils import (
    graph_statistics,
    degree_statistics,
    plot_degree_distribution
)

from src.split import split_edges

from src.negative_sampling import (
    generate_negative_edges,
    generate_query_negative_edges
)

from src.sanity_checks import (
    verify_positive_split,
    verify_negative_edges,
    analyze_isolated_nodes,
    verify_feature_source
)

from src.gnn import networkx_to_pyg

from src.baseline import (
    common_neighbors_scores,
    jaccard_scores,
    adamic_adar_scores,
    preferential_attachment_scores
)

from src.evaluation import (
    evaluate_scores,
    hits_at_k
)

from src.feature_ablation import (
    run_feature_ablation
)

from src.experiments import (
    run_gnn_multiseed
)

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "data/ca-AstroPh.txt.gz"

SEED = 42

TEST_RATIO = 0.10
VAL_RATIO = 0.10

# ------------------------------------------------------------
# GNN
# ------------------------------------------------------------

HIDDEN_CHANNELS = 64
OUT_CHANNELS = 32
DECODER_HIDDEN_DIM = 64

LEARNING_RATE = 0.001
EPOCHS = 500

DEVICE = "cpu"

# ------------------------------------------------------------
# HITS@K
# ------------------------------------------------------------

NUM_QUERY_POSITIVES = 1000
NEGATIVES_PER_QUERY = 999

HITS_K_VALUES = (
    10,
    50,
    100
)

# ------------------------------------------------------------
# EXPERIMENTS
# ------------------------------------------------------------

ABLATION_EPOCHS = 400
ABLATION_SEED = 1

GNN_SEEDS = (
    1,
    2,
    3,
    4,
    5
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# UTILITY
# ============================================================

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# BASELINE QUERY EVALUATION
# ============================================================

def evaluate_baseline_hits(
    G,
    query_positive_edges,
    query_negative_edges,
    scoring_function,
    name
):
    """
    Calcola Hits@K per un baseline utilizzando
    esattamente le query definite per l'esperimento.

    Gli score vengono calcolati su tutte le coppie
    candidate e poi ricostruiti nella struttura
    query-based richiesta da hits_at_k().
    """

    print()
    print(f"Computing Hits@K for {name}...")

    positive_scores = scoring_function(
        G,
        query_positive_edges
    )

    negative_scores = []

    for query_index, negative_edges in enumerate(
        query_negative_edges
    ):

        scores = scoring_function(
            G,
            negative_edges
        )

        negative_scores.append(
            scores
        )

        if (
            query_index + 1
        ) % 100 == 0:

            print(
                f"  Processed "
                f"{query_index + 1}/"
                f"{len(query_negative_edges)} queries"
            )

    results = {}

    for k in HITS_K_VALUES:

        results[k] = hits_at_k(
            positive_scores,
            negative_scores,
            k
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. LOAD GRAPH
    # ========================================================

    print_section(
        "LOADING GRAPH"
    )

    print(
        f"Dataset: {DATASET_PATH}"
    )

    G = load_graph(
        DATASET_PATH
    )

    # ========================================================
    # 2. GRAPH STATISTICS
    # ========================================================

    print_section(
        "GRAPH STATISTICS"
    )

    graph_stats = graph_statistics(
        G
    )

    degree_stats = degree_statistics(
        G
    )

    print(
        f"Nodes:       "
        f"{graph_stats['nodes']}"
    )

    print(
        f"Edges:       "
        f"{graph_stats['edges']}"
    )

    print(
        f"Density:     "
        f"{graph_stats['density']}"
    )

    print(
        f"Components:  "
        f"{graph_stats['connected_components']}"
    )

    print(
        f"Min degree:  "
        f"{degree_stats['min_degree']}"
    )

    print(
        f"Max degree:  "
        f"{degree_stats['max_degree']}"
    )

    print(
        f"Avg degree:  "
        f"{degree_stats['average_degree']}"
    )

    print(
        f"Median:      "
        f"{degree_stats['median_degree']}"
    )

    # ========================================================
    # 3. DEGREE DISTRIBUTION
    # ========================================================

    print_section(
        "DEGREE DISTRIBUTION"
    )

    plot_degree_distribution(
        G,
        "Degree Distribution - ca-AstroPh"
    )

    # ========================================================
    # 4. EDGE SPLIT
    # ========================================================

    print_section(
        "EDGE SPLIT"
    )

    train_graph, val_edges, test_edges = split_edges(
        G,
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED
    )

    train_edges = list(
        train_graph.edges()
    )

    print(
        f"Original edges:   "
        f"{G.number_of_edges()}"
    )

    print(
        f"Train edges:      "
        f"{len(train_edges)}"
    )

    print(
        f"Validation edges: "
        f"{len(val_edges)}"
    )

    print(
        f"Test edges:       "
        f"{len(test_edges)}"
    )

    # ========================================================
    # 5. NEGATIVE SAMPLING
    # ========================================================

    print_section(
        "NEGATIVE SAMPLING"
    )

    # Tutti i negativi devono essere disgiunti.
    original_edges = set(
        tuple(sorted(edge))
        for edge in G.edges()
    )

    train_negative_edges = generate_negative_edges(
        G,
        len(train_edges),
        seed=SEED,
        forbidden_edges=(
            original_edges
        )
    )

    val_negative_edges = generate_negative_edges(
        G,
        len(val_edges),
        seed=SEED + 1,
        forbidden_edges=(
            original_edges
            | {
                tuple(sorted(edge))
                for edge in train_negative_edges
            }
        )
    )

    test_negative_edges = generate_negative_edges(
        G,
        len(test_edges),
        seed=SEED + 2,
        forbidden_edges=(
            original_edges
            | {
                tuple(sorted(edge))
                for edge in train_negative_edges
            }
            | {
                tuple(sorted(edge))
                for edge in val_negative_edges
            }
        )
    )

    print(
        f"Train negatives:      "
        f"{len(train_negative_edges)}"
    )

    print(
        f"Validation negatives: "
        f"{len(val_negative_edges)}"
    )

    print(
        f"Test negatives:       "
        f"{len(test_negative_edges)}"
    )

    # ========================================================
    # 6. SANITY CHECKS
    # ========================================================

    print_section(
        "SANITY CHECKS"
    )

    positive_valid = verify_positive_split(
        G,
        train_graph,
        val_edges,
        test_edges
    )

    negative_valid = verify_negative_edges(
        G,
        train_graph,
        train_negative_edges,
        val_negative_edges,
        test_negative_edges,
        val_edges,
        test_edges
    )

    analyze_isolated_nodes(
        train_graph,
        val_edges,
        test_edges
    )

    if not positive_valid:
        raise RuntimeError(
            "Positive split non valido."
        )

    if not negative_valid:
        raise RuntimeError(
            "Negative sampling non valido."
        )

    # ========================================================
    # 7. HITS QUERY GENERATION
    # ========================================================

    print_section(
        "HITS@K QUERY GENERATION"
    )

    if len(test_edges) < NUM_QUERY_POSITIVES:

        raise ValueError(
            "Il test set contiene meno archi "
            "delle query richieste."
        )

    query_positive_edges = list(
        test_edges[:NUM_QUERY_POSITIVES]
    )

    query_negative_edges = (
        generate_query_negative_edges(
            G,
            query_positive_edges,
            num_negatives_per_query=(
                NEGATIVES_PER_QUERY
            ),
            seed=SEED
        )
    )

    print(
        f"Queries: "
        f"{len(query_positive_edges)}"
    )

    print(
        f"Negative candidates/query: "
        f"{NEGATIVES_PER_QUERY}"
    )

    print(
        f"Candidates/query: "
        f"{1 + NEGATIVES_PER_QUERY}"
    )

    print(
        f"Total negative candidates: "
        f"{sum(len(x) for x in query_negative_edges)}"
    )

    # ========================================================
    # 8. BASELINES
    # ========================================================

    print_section(
        "BASELINE LINK PREDICTION"
    )

    baseline_methods = {

        "Common Neighbors":
            common_neighbors_scores,

        "Jaccard":
            jaccard_scores,

        "Adamic-Adar":
            adamic_adar_scores,

        "Preferential Attachment":
            preferential_attachment_scores
    }

    baseline_results = {}

    for name, scoring_function in (
        baseline_methods.items()
    ):

        print()
        print("-" * 70)
        print(name)
        print("-" * 70)

        positive_scores = scoring_function(
            train_graph,
            test_edges
        )

        negative_scores = scoring_function(
            train_graph,
            test_negative_edges
        )

        metrics = evaluate_scores(
            positive_scores,
            negative_scores
        )

        print(
            f"Test AUC: "
            f"{metrics['auc']:.6f}"
        )

        print(
            f"Test AP:  "
            f"{metrics['ap']:.6f}"
        )

        baseline_results[name] = {
            "auc": metrics["auc"],
            "ap": metrics["ap"]
        }

    # ========================================================
    # 9. BASELINE HITS@K
    # ========================================================

    print_section(
        "BASELINE HITS@K"
    )

    baseline_hits = {}

    for name, scoring_function in (
        baseline_methods.items()
    ):

        hits = evaluate_baseline_hits(
            train_graph,
            query_positive_edges,
            query_negative_edges,
            scoring_function,
            name
        )

        baseline_hits[name] = hits

        print()
        print(name)

        for k in HITS_K_VALUES:

            print(
                f"Hits@{k}: "
                f"{hits[k]:.4f}"
            )

    # ========================================================
    # 10. PYTORCH GEOMETRIC DATA
    # ========================================================

    print_section(
        "GNN DATA"
    )

    data, node_to_idx = networkx_to_pyg(
        train_graph,
        feature_mode="full"
    )

    data = data.to(
        DEVICE
    )

    print(
        f"Nodes: "
        f"{data.num_nodes}"
    )

    print(
        f"Node features: "
        f"{data.x.shape[1]}"
    )

    print(
        f"Message-passing edges: "
        f"{data.edge_index.shape[1]}"
    )

    # ========================================================
    # 11. FEATURE SOURCE CHECK
    # ========================================================

    verify_feature_source(
        G,
        train_graph,
        data
    )

    # ========================================================
    # 12. FEATURE ABLATION
    # ========================================================

    feature_ablation_results = (
        run_feature_ablation(
            train_graph=train_graph,
            train_edges=train_edges,
            train_negative_edges=(
                train_negative_edges
            ),
            val_edges=val_edges,
            val_negative_edges=(
                val_negative_edges
            ),
            test_edges=test_edges,
            test_negative_edges=(
                test_negative_edges
            ),
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=OUT_CHANNELS,
            decoder_hidden_dim=(
                DECODER_HIDDEN_DIM
            ),
            learning_rate=0.01,
            epochs=ABLATION_EPOCHS,
            seed=ABLATION_SEED,
            device=DEVICE
        )
    )

    # ========================================================
    # 13. GNN MULTI-SEED
    # ========================================================

    print_section(
        "GNN MULTI-SEED EXPERIMENT"
    )

    gnn_results = run_gnn_multiseed(
        train_graph=train_graph,
        train_edges=train_edges,
        train_negative_edges=(
            train_negative_edges
        ),
        val_edges=val_edges,
        val_negative_edges=(
            val_negative_edges
        ),
        test_edges=test_edges,
        test_negative_edges=(
            test_negative_edges
        ),
        query_positive_edges=(
            query_positive_edges
        ),
        query_negative_edges=(
            query_negative_edges
        ),
        seeds=GNN_SEEDS,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
        decoder_hidden_dim=(
            DECODER_HIDDEN_DIM
        ),
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )

    # ========================================================
    # 14. FINAL COMPARISON
    # ========================================================

    print_section(
        "FINAL COMPARISON"
    )

    print()
    print(
        f"{'Method':<25}"
        f"{'AUC':>12}"
        f"{'AP':>12}"
        f"{'Hits@10':>12}"
        f"{'Hits@50':>12}"
        f"{'Hits@100':>12}"
    )

    print("-" * 75)

    for name in baseline_methods:

        result = baseline_results[
            name
        ]

        hits = baseline_hits[
            name
        ]

        print(
            f"{name:<25}"
            f"{result['auc']:>12.4f}"
            f"{result['ap']:>12.4f}"
            f"{hits[10]:>12.4f}"
            f"{hits[50]:>12.4f}"
            f"{hits[100]:>12.4f}"
        )

    # --------------------------------------------------------
    # GNN
    # --------------------------------------------------------

    if len(gnn_results) > 0:

        gnn_test_auc = np.mean([
            result["test_auc"]
            for result in gnn_results
        ])

        gnn_test_ap = np.mean([
            result["test_ap"]
            for result in gnn_results
        ])

        gnn_hits_10 = np.mean([
            result["hits@10"]
            for result in gnn_results
        ])

        gnn_hits_50 = np.mean([
            result["hits@50"]
            for result in gnn_results
        ])

        gnn_hits_100 = np.mean([
            result["hits@100"]
            for result in gnn_results
        ])

        print(
            f"{'GraphSAGE + MLP':<25}"
            f"{gnn_test_auc:>12.4f}"
            f"{gnn_test_ap:>12.4f}"
            f"{gnn_hits_10:>12.4f}"
            f"{gnn_hits_50:>12.4f}"
            f"{gnn_hits_100:>12.4f}"
        )

    # ========================================================
    # 15. DONE
    # ========================================================

    print()
    print("=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()