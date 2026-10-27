"""
MODULO PER L'ORCHESTRAZIONE DEI CONTROLLI DI INTERGRITA' DELLA PIPELINE
--> Raccoglie tutti i test di sanity check e blocca immediatamente l'esecuzione
sollevando un RunTimeError.
"""

from src.utils.sanity_checks import (
    verify_gnn_split,
    verify_positive_split,
    verify_negative_edges,
    analyze_isolated_nodes,
)


def run_sanity_checks(graph, splits, negatives, print_section):
    """
    Esegue in sequenza tutti i controlli sui set di dati generici 
    """

    print_section("SANITY CHECKS")

    train_graph = splits["train_graph"]
    message_passing_graph = splits["message_passing_graph"]
    message_passing_edges = splits["message_passing_edges"]
    supervised_train_edges = splits["supervised_train_edges"]
    val_edges = splits["val_edges"]
    test_edges = splits["test_edges"]

    gnn_split_valid = verify_gnn_split(
        train_graph=train_graph,
        message_passing_graph=message_passing_graph,
        message_passing_edges=message_passing_edges,
        supervised_train_edges=supervised_train_edges,
        supervised_train_negative_edges=negatives[
            "supervised_train_negative_edges"
        ],
        val_negative_edges=negatives["val_negative_edges"],
        test_negative_edges=negatives["test_negative_edges"],
        val_edges=val_edges,
        test_edges=test_edges,
    )

    if not gnn_split_valid:
        raise RuntimeError("Controlli dello split GNN falliti.")

    positive_valid = verify_positive_split(
        graph,
        train_graph,
        val_edges,
        test_edges,
    )

    negative_valid = verify_negative_edges(
        graph,
        train_graph,
        negatives["train_negative_edges"],
        negatives["val_negative_edges"],
        negatives["test_negative_edges"],
        val_edges,
        test_edges,
    )

    analyze_isolated_nodes(train_graph, val_edges, test_edges)
    analyze_isolated_nodes(message_passing_graph, val_edges, test_edges, graph_name="message_passing")

    if not positive_valid:
        raise RuntimeError("Positive split non valido.")

    if not negative_valid:
        raise RuntimeError("Negative sampling non valido.")

