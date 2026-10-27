"""
ENTRY POINT DEL PROGETTO

Lo script orchestra l'intera pipeline sperimentale per la Link Prediction.
Le operazioni che vengono svolte sono:
    - caricamento e ispezione del dataset;
    - suddivisione dei dati e negative sampling;
    - controlli anti-leakage;
    - valutazione euristiche classiche;
    - valutazione GNN.
"""

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
    """
    Gestisce gli argomenti da riga di comando(CLI) e permette di eseguire
    selettivamente solo una parte degli esperimenti, risparmiando tempo durante 
    la fase di testing e debugging.
    """
    parser = argparse.ArgumentParser(description="Pipeline di link prediction con baseline e GNN.")

    parser.add_argument(
        "--run",
        choices=("all", "baselines", "gnn"),
        default="all",
        help="Esperimenti da eseguire (default: all).",
    )
    return parser.parse_args()


#funzione principale che esegue l'intera metadologia della tesi
def main():
    
    args = parse_args()


    """
    blocco assoluto dei generatori casuali per garantire che
    l'esperimento sia matematicamente riproducibile al 100%
    """
    set_global_seed(SEED)

    # caricamento del dataset ed esplorazione
    print_section("LOADING GRAPH")
    print(f"Dataset: {DATASET_PATH}")

    graph = load_graph(DATASET_PATH)

    print_section("GRAPH STATISTICS")
    print_graph_summary(graph, graph_statistics, degree_statistics)

    print_section("DEGREE DISTRIBUTION")

    #genera il grafo a coda lunga tipico delle reti sociali
    plot_degree_distribution(graph, f"Degree Distribution - {DATASET_NAME}")



    """
    suddivide il grafo garantendo che nessun arco di test/val 
    venga utilizzato per l'addestramento e genera i distrattori(negative sampling)
    """
    splits = prepare_graph_splits(graph)
    negatives = prepare_negative_edges(graph, splits)


    # assicura matematicamente che non ci siano sovrapposizioni illecite
    run_sanity_checks(graph, splits, negatives, print_section)

    # crea i set candidati per il calcolo di hits@K
    queries = prepare_hits_queries(graph, splits["test_edges"])

    # esperimenti sulle baseline tradizionali nelle modalità 'all' e 'baselines'
    if args.run in ("all", "baselines"):
        # esegue le formule matematiche su entrambi i protocolli:
        #   --> train
        #   --> message-passing
        baseline_results = run_baseline_experiments(
            splits=splits,
            negatives=negatives,
            queries=queries,
            print_section=print_section,
        )

        #analisi del degree bias
        run_additional_baseline_analyses(
            splits=splits,
            negatives=negatives,
            methods=baseline_results["methods"]
        )

        # stampa dei risultati finali per le baseline
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

    # esperimenti con graph neural network nelle modalità 'all' e 'gnn'
    if args.run in ("all", "gnn"):
        # converte il grafo in tensore per la scheda video
        data = prepare_gnn_data(
            splits["message_passing_graph"],
            print_section=print_section,
        )

        # lancia lo studio di ablazione e il training multi-seed
        gnn_experiments = run_gnn_experiments(
            splits,
            negatives,
            queries,
            graph,
            data,
            print_section=print_section,
        )

        # esporta report in CSV
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
