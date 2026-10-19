import argparse

from config import DATASET_PATH, SEED, DATASET_NAME

from src.reporting import (
    print_section,
    print_graph_summary,
    print_baseline_results,
)
from src.reproducibility import set_global_seed
from src.pipeline_data import (
    prepare_graph_splits,
    prepare_negative_edges,
    prepare_hits_queries,
)
from src.utils.pipeline_checks import run_sanity_checks
from src.data_loader import load_graph
from src.graph_utils import (
    graph_statistics,
    degree_statistics,
    plot_degree_distribution,
)
from experiments.run_baselines import (
    run_baseline_experiments,
    run_additional_baseline_analyses,
)
from experiments.run_gnn import (
    prepare_gnn_data,
    run_gnn_experiments,
    save_gnn_results,
)


def parse_args():
    """Legge la modalità di esecuzione richiesta."""
    parser = argparse.ArgumentParser(
        description="Pipeline di link prediction con baseline e GNN."
    )
    parser.add_argument(
        "--run",
        choices=("all", "baselines", "gnn"),
        default="all",
        help="Esperimenti da eseguire (default: all).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    set_global_seed(SEED)

    # 1. Load and inspect the graph.
    print_section("LOADING GRAPH")
    print(f"Dataset: {DATASET_PATH}")

    graph = load_graph(DATASET_PATH)

    print_section("GRAPH STATISTICS")
    print_graph_summary(graph, graph_statistics, degree_statistics)

    print_section("DEGREE DISTRIBUTION")
    plot_degree_distribution(graph, f"Degree Distribution - {DATASET_NAME}")

    # 2. Split positive edges and generate negative examples.
    splits = prepare_graph_splits(graph)
    negatives = prepare_negative_edges(graph, splits)

    # 3. Validate the data protocol before experiments.
    run_sanity_checks(graph, splits, negatives, print_section)

    # 4. Prepare the fixed Hits@K candidate sets.
    queries = prepare_hits_queries(graph, splits["test_edges"])

    # 5. Run baseline experiments when requested.
    if args.run in ("all", "baselines"):
        baseline_results = run_baseline_experiments(
            splits=splits,
            negatives=negatives,
            queries=queries,
            print_section=print_section,
        )

        run_additional_baseline_analyses(
            splits=splits,
            negatives=negatives,
            methods=baseline_results["methods"],
            print_section=print_section,
        )

        print_baseline_results(
            "FINAL COMPARISON - BASELINES ON TRAIN GRAPH",
            baseline_results["train_results"],
            baseline_results["train_hits"],
        )

        print_baseline_results(
            "FINAL COMPARISON - BASELINES ON MESSAGE-PASSING GRAPH",
            baseline_results["message_passing_results"],
            baseline_results["message_passing_hits"],
        )

    # 6. Run GNN experiments when requested.
    if args.run in ("all", "gnn"):
        data = prepare_gnn_data(
            splits["message_passing_graph"],
            print_section=print_section,
        )

        gnn_experiments = run_gnn_experiments(
            splits,
            negatives,
            queries,
            graph,
            data,
            print_section=print_section,
        )

        save_gnn_results(
            gnn_experiments["decoder_comparison"],
            print_section,
        )

    print()
    print("=" * 70)
    print(f"PIPELINE COMPLETED - MODE: {args.run.upper()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
