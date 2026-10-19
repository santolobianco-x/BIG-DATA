from experiments.baseline_setup import get_baseline_methods
from experiments.baseline_evaluation import evaluate_baselines
from experiments.run_degree_bias import run_degree_bias_experiment

from config import RESULTS_DIR, HITS_K_VALUES, DATASET_NAME
from experiments.export_results import (
    export_rows,
    flatten_degree_bias_results,
)


def run_baseline_experiments(splits, negatives, queries, print_section):
    """Esegue le baseline sui due grafi previsti dal protocollo."""

    baseline_methods = get_baseline_methods()

    query_positive_edges, query_negative_edges = queries
    test_edges = splits["test_edges"]
    test_negative_edges = negatives["test_negative_edges"]

    # Protocollo originale: train_graph.
    baseline_results, baseline_hits = evaluate_baselines(
        graph=splits["train_graph"],
        test_edges=test_edges,
        test_negative_edges=test_negative_edges,
        query_positive_edges=query_positive_edges,
        query_negative_edges=query_negative_edges,
        baseline_methods=baseline_methods,
        title="BASELINES - TRAIN GRAPH",
        print_section=print_section,
    )

    # Protocollo alternativo: message_passing_graph.
    (
        message_passing_baseline_results,
        message_passing_baseline_hits,
    ) = evaluate_baselines(
        graph=splits["message_passing_graph"],
        test_edges=test_edges,
        test_negative_edges=test_negative_edges,
        query_positive_edges=query_positive_edges,
        query_negative_edges=query_negative_edges,
        baseline_methods=baseline_methods,
        title="BASELINES - MESSAGE-PASSING GRAPH",
        print_section=print_section,
    )


    baseline_rows = []

    for graph_name, results, hits_results in [
        (
            "train_graph",
            baseline_results,
            baseline_hits,
        ),
        (
            "message_passing_graph",
            message_passing_baseline_results,
            message_passing_baseline_hits,
        ),
    ]:
        for baseline, metrics in results.items():
            row = {
                "graph": graph_name,
                "baseline": baseline,
                "auc": metrics["auc"],
                "ap": metrics["ap"],
            }

            for k in HITS_K_VALUES:
                row[f"hits@{k}"] = hits_results[baseline][k]

            baseline_rows.append(row)

    export_rows(
        baseline_rows,
        RESULTS_DIR / f"{DATASET_NAME}_baseline_results.csv",
    )

    return {
        "methods": baseline_methods,
        "train_results": baseline_results,
        "train_hits": baseline_hits,
        "message_passing_results": message_passing_baseline_results,
        "message_passing_hits": message_passing_baseline_hits,
    }



def run_additional_baseline_analyses(
    splits,
    negatives,
    methods,
    print_section,
):

    degree_bias_results = run_degree_bias_experiment(
        train_graph=splits["train_graph"],
        positive_edges=splits["test_edges"],
        negative_edges=negatives["test_negative_edges"],
        scoring_functions=methods,
    )

    export_rows(
        flatten_degree_bias_results(degree_bias_results),
        RESULTS_DIR / f"{DATASET_NAME}_degree_bias_results.csv",
    )

    return degree_bias_results