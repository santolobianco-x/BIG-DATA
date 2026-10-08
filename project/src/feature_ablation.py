import torch

from src.gnn import (
    networkx_to_pyg,
    GraphSAGE,
    MLPDecoder,
    train_gnn,
    prepare_edges,
    predict_edges
)



from src.evaluation import evaluate_scores


def run_feature_ablation(
    train_graph,
    train_edges,
    train_negative_edges,
    val_edges,
    val_negative_edges,
    test_edges,
    test_negative_edges,
    hidden_channels=64,
    out_channels=32,
    decoder_hidden_dim=64,
    learning_rate=0.01,
    epochs=400,
    seed=1,
    device="cpu"
):
    """
    Esegue il Feature Ablation Study.

    Confronta quattro configurazioni di feature:

        - full
        - degree
        - clustering
        - constant

    Tutti gli altri aspetti dell'esperimento rimangono
    invariati.

    Parameters
    ----------
    train_graph : networkx.Graph
        Grafo utilizzato per il message passing.

    train_edges : list
        Archi positivi di training.

    train_negative_edges : list
        Archi negativi di training.

    val_edges : list
        Archi positivi di validation.

    val_negative_edges : list
        Archi negativi di validation.

    test_edges : list
        Archi positivi di test.

    test_negative_edges : list
        Archi negativi di test.

    hidden_channels : int
        Dimensione dello spazio nascosto di GraphSAGE.

    out_channels : int
        Dimensione degli embedding finali.

    decoder_hidden_dim : int
        Dimensione dello strato nascosto del decoder.

    learning_rate : float
        Learning rate dell'ottimizzatore.

    epochs : int
        Numero di epoche.

    seed : int
        Seed utilizzato per la riproducibilità.

    device : str
        Device utilizzato per il training.

    Returns
    -------
    dict
        Risultati dell'ablation study.
    """

    feature_modes = [
        "full",
        "degree",
        "clustering",
        "constant"
    ]

    results = {}

    print("\n" + "=" * 70)
    print("FEATURE ABLATION STUDY")
    print("=" * 70)

    for mode in feature_modes:

        print("\n" + "-" * 70)
        print(f"FEATURE CONFIGURATION: {mode.upper()}")
        print("-" * 70)

        # ----------------------------------------------------
        # 1. CREAZIONE DEL GRAFO PYTORCH GEOMETRIC
        # ----------------------------------------------------

        data, node_to_idx = networkx_to_pyg(
            train_graph,
            feature_mode=mode
        )

        data = data.to(device)

        print(
            f"Input features: "
            f"{data.x.shape[1]}"
        )

        print(
            f"Nodes: "
            f"{data.num_nodes}"
        )

        print(
            f"Message-passing edges: "
            f"{data.edge_index.shape[1]}"
        )

        # ----------------------------------------------------
        # 2. CREAZIONE DEL MODELLO
        # ----------------------------------------------------

        model = GraphSAGE(
            in_channels=data.x.shape[1],
            hidden_channels=hidden_channels,
            out_channels=out_channels
        ).to(device)

        decoder = MLPDecoder(
            embedding_dim=out_channels,
            hidden_dim=decoder_hidden_dim
        ).to(device)

        # ----------------------------------------------------
        # 3. OPTIMIZER
        # ----------------------------------------------------

        optimizer = torch.optim.Adam(
            list(model.parameters())
            + list(decoder.parameters()),
            lr=learning_rate
        )

        # ----------------------------------------------------
        # 4. TRAINING
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
        # 5. TEST EDGE CONVERSION
        # ----------------------------------------------------

        test_edge_index, test_labels = prepare_edges(
            test_edges,
            test_negative_edges,
            node_to_idx
        )

        test_edge_index = test_edge_index.to(device)

        test_labels = test_labels.to(device)

        # ----------------------------------------------------
        # 6. TEST PREDICTION
        # ----------------------------------------------------

        test_probabilities = predict_edges(
            model=model,
            decoder=decoder,
            data=data,
            edge_index=test_edge_index
        )

        # ----------------------------------------------------
        # 7. SEPARAZIONE POSITIVI / NEGATIVI
        # ----------------------------------------------------

        n_positive = len(test_edges)

        positive_scores = (
            test_probabilities[:n_positive]
            .detach()
            .cpu()
            .numpy()
        )

        negative_scores = (
            test_probabilities[n_positive:]
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # 8. TEST METRICS
        # ----------------------------------------------------

        test_metrics = evaluate_scores(
            positive_scores,
            negative_scores
        )

        test_auc = test_metrics["auc"]
        test_ap = test_metrics["ap"]

        # ----------------------------------------------------
        # 9. SALVATAGGIO RISULTATI
        # ----------------------------------------------------

        results[mode] = {
            "input_features": data.x.shape[1],
            "best_epoch": history["best_epoch"],
            "val_auc": history["best_val_auc"],
            "val_ap": history["best_val_ap"],
            "test_auc": test_auc,
            "test_ap": test_ap
        }

        # ----------------------------------------------------
        # 10. OUTPUT
        # ----------------------------------------------------

        print()
        print(
            f"Best epoch:       "
            f"{history['best_epoch']}"
        )

        print(
            f"Validation AUC:   "
            f"{history['best_val_auc']:.4f}"
        )

        print(
            f"Validation AP:    "
            f"{history['best_val_ap']:.4f}"
        )

        print(
            f"Test AUC:         "
            f"{test_auc:.4f}"
        )

        print(
            f"Test AP:          "
            f"{test_ap:.4f}"
        )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print("\n" + "=" * 70)
    print("FEATURE ABLATION RESULTS")
    print("=" * 70)

    print(
        f"{'Feature':<15}"
        f"{'Val AUC':>12}"
        f"{'Val AP':>12}"
        f"{'Test AUC':>12}"
        f"{'Test AP':>12}"
    )

    print("-" * 70)

    for mode in feature_modes:

        result = results[mode]

        print(
            f"{mode:<15}"
            f"{result['val_auc']:>12.4f}"
            f"{result['val_ap']:>12.4f}"
            f"{result['test_auc']:>12.4f}"
            f"{result['test_ap']:>12.4f}"
        )

    print("=" * 70)

    return results