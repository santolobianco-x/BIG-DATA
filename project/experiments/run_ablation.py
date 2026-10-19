
import torch
import numpy as np

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
    message_passing_graph,
    supervised_train_edges,
    supervised_train_negative_edges,
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
    Confronta quattro configurazioni di feature:
    - full
    - degree
    - clustering
    - constant

    L'encoder utilizza soltanto il message-passing graph.
    Il decoder viene addestrato sugli archi supervisionati.
    Il checkpoint migliore viene selezionato sulla validation.
    Il test viene utilizzato esclusivamente per la valutazione.
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
        # 1. RIPRODUCIBILITÀ
        # ----------------------------------------------------

        torch.manual_seed(seed)
        np.random.seed(seed)

        # ----------------------------------------------------
        # 2. CREAZIONE DEL GRAFO PYTORCH GEOMETRIC
        # ----------------------------------------------------

        data, node_to_idx = networkx_to_pyg(
            message_passing_graph,
            feature_mode=mode
        )

        data = data.to(device)

        if mode == "constant":
            degrees = torch.bincount(
                data.edge_index[0],
                minlength=data.num_nodes
            )

            print(
                "Feature rows uniche:",
                torch.unique(data.x, dim=0).size(0)
            )
            print(
                "Nodi isolati nel message-passing graph:",
                (degrees == 0).sum().item()
            )
            print(
                "Nodi non isolati:",
                (degrees > 0).sum().item()
            )

        print(f"Input features: {data.x.shape[1]}")
        print(f"Nodes: {data.num_nodes}")
        print(
            "Message-passing edges: "
            f"{data.edge_index.shape[1]}"
        )
        print(
            "Supervised positive edges: "
            f"{len(supervised_train_edges)}"
        )
        print(
            "Supervised negative edges: "
            f"{len(supervised_train_negative_edges)}"
        )

        # ----------------------------------------------------
        # 3. CREAZIONE DEL MODELLO
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
        # 4. OPTIMIZER
        # ----------------------------------------------------

        optimizer = torch.optim.Adam(
            list(model.parameters())
            + list(decoder.parameters()),
            lr=learning_rate
        )

        # ----------------------------------------------------
        # 5. TRAINING
        # ----------------------------------------------------

        history = train_gnn(
            model=model,
            decoder=decoder,
            data=data,
            train_edges=supervised_train_edges,
            train_negative_edges=supervised_train_negative_edges,
            val_edges=val_edges,
            val_negative_edges=val_negative_edges,
            node_to_idx=node_to_idx,
            optimizer=optimizer,
            epochs=epochs,
            keep_ratio=1.0,
            seed=seed
        )

        # ----------------------------------------------------
        # 6. CONVERSIONE DEGLI ARCHI DI TEST
        # ----------------------------------------------------

        test_edge_index, _ = prepare_edges(
            test_edges,
            test_negative_edges,
            node_to_idx
        )

        test_edge_index = test_edge_index.to(device)

        # ----------------------------------------------------
        # 7. PREDIZIONI SUL TEST
        # ----------------------------------------------------

        model.eval()
        decoder.eval()

        test_scores = predict_edges(
            model=model,
            decoder=decoder,
            data=data,
            edge_index=test_edge_index
        )

        # ----------------------------------------------------
        # 8. SEPARAZIONE POSITIVI E NEGATIVI
        # ----------------------------------------------------

        n_positive = len(test_edges)

        positive_scores = (
            test_scores[:n_positive]
            .detach()
            .cpu()
            .numpy()
        )

        negative_scores = (
            test_scores[n_positive:]
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # 9. DIAGNOSTICA APPROFONDITA: FEATURE CONSTANT
        # ----------------------------------------------------

        if mode == "constant":

            with torch.no_grad():
                z = model(data.x, data.edge_index)

            degrees = torch.bincount(
                data.edge_index[0],
                minlength=data.num_nodes
            )

            isolated = degrees == 0
            non_isolated = degrees > 0

            z_isolated = z[isolated]
            z_non_isolated = z[non_isolated]

            print("\n" + "=" * 60)
            print("CONSTANT FEATURE DIAGNOSTICS")
            print("=" * 60)

            print("Embedding shape:", tuple(z.shape))

            print(
                "Unique embeddings exact:",
                torch.unique(z, dim=0).size(0)
            )

            print(
                "Unique embeddings rounded 6dp:",
                torch.unique(
                    z.round(decimals=6),
                    dim=0
                ).size(0)
            )

            print(
                "Embedding std complessiva:",
                z.std().item()
            )

            if z_non_isolated.size(0) > 0:
                print(
                    "Max difference non-isolated:",
                    (
                        z_non_isolated - z_non_isolated[0]
                    ).abs().max().item()
                )

                print(
                    "Std embedding nodi non isolati:",
                    z_non_isolated.std().item()
                )

            if z_isolated.size(0) > 0:
                print(
                    "Std embedding nodi isolati:",
                    z_isolated.std().item()
                )

                print(
                    "Max absolute isolated embedding:",
                    z_isolated.abs().max().item()
                )

            print("\nPositive scores:")
            print("  Min:", float(np.min(positive_scores)))
            print("  Max:", float(np.max(positive_scores)))
            print("  Mean:", float(np.mean(positive_scores)))
            print("  Std:", float(np.std(positive_scores)))

            print("\nNegative scores:")
            print("  Min:", float(np.min(negative_scores)))
            print("  Max:", float(np.max(negative_scores)))
            print("  Mean:", float(np.mean(negative_scores)))
            print("  Std:", float(np.std(negative_scores)))

            print(
                "\nPositive scores finite:",
                bool(np.isfinite(positive_scores).all())
            )
            print(
                "Negative scores finite:",
                bool(np.isfinite(negative_scores).all())
            )

        # ----------------------------------------------------
        # 10. METRICHE DI TEST
        # ----------------------------------------------------

        test_metrics = evaluate_scores(
            positive_scores,
            negative_scores
        )

        test_auc = test_metrics["auc"]
        test_ap = test_metrics["ap"]

        # ----------------------------------------------------
        # 11. SALVATAGGIO RISULTATI
        # ----------------------------------------------------

        results[mode] = {
            "input_features": data.x.shape[1],
            "best_epoch": history["best_epoch"],
            "epochs_trained": history["epochs_trained"],
            "val_auc": history["best_val_auc"],
            "val_ap": history["best_val_ap"],
            "test_auc": test_auc,
            "test_ap": test_ap
        }

        # ----------------------------------------------------
        # 12. OUTPUT
        # ----------------------------------------------------

        print()
        print(f"Best epoch:       {history['best_epoch']}")
        print(f"Validation AUC:   {history['best_val_auc']:.4f}")
        print(f"Validation AP:    {history['best_val_ap']:.4f}")
        print(f"Test AUC:         {test_auc:.4f}")
        print(f"Test AP:          {test_ap:.4f}")

    # ========================================================
    # TABELLA FINALE
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


