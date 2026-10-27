"""
MODULO PER LO STUDIO DI ABLAZIONE SULLE FEATURE DEI NODI

--> testa la GNN asportando le informazioni di partenza, per isolare 
    quale informazione guida davvero la performance della rete.
"""
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
    Esegue addestramento e valutazione della GNN usando 
    4 configurazioni diverse. Le configurazioni sono:
    - full --> configurazione completa di riferimento;
    - degree --> informazione sul numero di vicini del nodo;
    - clustering --> informazione che descrive quanto i vicini
            di un nodo siano collegati tra loro
    - constant --> assenza di informazioni iniziali che distinguano i nodi
    """

    feature_modes = ["full", "degree", "clustering","constant"]

    results = {}

    print("\n" + "=" * 70)
    print("FEATURE ABLATION STUDY")
    print("=" * 70)

    # esegue il ciclo per ogni configurazione
    # ricrea i dati, inizializza un modello, lo allena e lo valuta
    for mode in feature_modes:

        print("\n" + "-" * 70)
        print(f"FEATURE CONFIGURATION: {mode.upper()}")
        print("-" * 70)

        # azzeramento seed per ogni run
        torch.manual_seed(seed)
        np.random.seed(seed)

        # creazione del grafo pythorch geometric
        data, node_to_idx = networkx_to_pyg(message_passing_graph,feature_mode=mode)
        data = data.to(device)

        if mode == "constant":
            degrees = torch.bincount(data.edge_index[0],minlength=data.num_nodes)

            # controlla se tutti i nodi hanno le stesse feature iniziali
            print( "Feature rows uniche:", torch.unique(data.x, dim=0).size(0))

            # analizza quanti nodi sono isolati e quanti no
            print( "Nodi isolati nel message-passing graph:", (degrees == 0).sum().item())
            print("Nodi non isolati:",(degrees > 0).sum().item())


        print(f"Input features: {data.x.shape[1]}")
        print(f"Nodes: {data.num_nodes}")
        print(f"Message-passing edges: {data.edge_index.shape[1]}")
        print(f"Supervised positive edges: {len(supervised_train_edges)}")
        print(f"Supervised negative edges: {len(supervised_train_negative_edges)}")

        # creazione del modello
        model = GraphSAGE(
            in_channels=data.x.shape[1], # numero feature di ciascun nodo
            hidden_channels=hidden_channels, # dimensione dello strato nascosto
            out_channels=out_channels # embedding finale di ogni nodo
        ).to(device)

        # creazione del decoder MLP
        decoder = MLPDecoder(
            embedding_dim=out_channels,
            hidden_dim=decoder_hidden_dim
        ).to(device)

        # adam deve aggiornare sia i parametri dell'encoder e 
        # sia i parametri del decoder
        # model.parameters() restituisce i parametri di GraphSAGE
        # decoder.parameters() restituisce quelli dell'MLP
        optimizer = torch.optim.Adam(
            list(model.parameters())
            + list(decoder.parameters()),
            lr=learning_rate
        )

        # si avvia l'addestramento

        # train_gnn gestisce il ciclo sulle epoche, calcola la loss, 
        # gli aggiornamenti dei pesi e la valutazione sul validation set

        # keep_ratio=1.0 permette di mantenere tutti gli archi disponibili
        # per il message passing 

        # history conterrà tutte le informazioni sul training
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

        
        test_edge_index, _ = prepare_edges(
            test_edges,
            test_negative_edges,
            node_to_idx
        )

        test_edge_index = test_edge_index.to(device)

        # imposta encoder e decoder in modalità valutazione
        model.eval()
        decoder.eval()

        # si calcolano i punteggi predetti per tutte le coppie di test
        test_scores = predict_edges(
            model=model,
            decoder=decoder,
            data=data,
            edge_index=test_edge_index
        )

        n_positive = len(test_edges)

        # separazione score positivi da score negativi
        positive_scores = (test_scores[:n_positive].detach().cpu().numpy())
        negative_scores = (test_scores[n_positive:].detach().cpu().numpy())


        
        if mode == "constant":

            # si disattiva il calcolo dei gradienti perchè si vogliono osservare
            # gli embedding

            with torch.no_grad():
                z = model(data.x, data.edge_index)

            # si riconta il grado dei nodi del grafo di message passing
            degrees = torch.bincount(
                data.edge_index[0],
                minlength=data.num_nodes
            )

            # maschera nodi isolati
            isolated = degrees == 0

            # maschera nodi non isolati
            non_isolated = degrees > 0

            z_isolated = z[isolated]
            z_non_isolated = z[non_isolated]


            print("\n" + "=" * 60)
            print("CONSTANT FEATURE DIAGNOSTICS")
            print("=" * 60)

            print("Embedding shape:", tuple(z.shape))

            # numero di embedding esattamente distinti
            print("Unique embeddings exact:", torch.unique(z, dim=0).size(0))


            """
            si ripete il conteggio dopo aver arrotondato i valori 
            a sei cifre decimali, serve a vedere se embedding 
            leggeremente diversi diventano uguali
            a precisione ridotta
            """
            print(
                "Unique embeddings rounded 6dp:",
                torch.unique(z.round(decimals=6),dim=0).size(0)
            )

            # deviazione standard di tutti gli elementi della matrice
            print("Embedding std complessiva:", z.std().item())

            # si controlla se esiste almeno un nodo di questo tipo e si
            # analizza l'embending dei nodi non isolati


            if z_non_isolated.size(0) > 0:
                # 0 significa che tutti coincidono esattamente con lo stesso embedding
                print("Max difference non-isolated:",
                    (z_non_isolated - z_non_isolated[0]).abs().max().item()
                )
                
                print("Std embedding nodi non isolati:", z_non_isolated.std().item())

            
            if z_isolated.size(0) > 0:
                print( "Std embedding nodi isolati:",z_isolated.std().item())
                print( "Max absolute isolated embedding:", z_isolated.abs().max().item())

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

        # si calcolano le metriche confrontando i punteggi degli archi
        # positivi con quelli degli archi negativi
        test_metrics = evaluate_scores(
            positive_scores,
            negative_scores
        )

        test_auc = test_metrics["auc"]
        test_ap = test_metrics["ap"]

        # dizionario associato alla configurazione corrente
        results[mode] = {
            "input_features": data.x.shape[1],
            "best_epoch": history["best_epoch"],
            "epochs_trained": history["epochs_trained"],
            "val_auc": history["best_val_auc"],
            "val_ap": history["best_val_ap"],
            "test_auc": test_auc,
            "test_ap": test_ap
        }


        print()
        print(f"Best epoch:       {history['best_epoch']}")
        print(f"Validation AUC:   {history['best_val_auc']:.4f}")
        print(f"Validation AP:    {history['best_val_ap']:.4f}")
        print(f"Test AUC:         {test_auc:.4f}")
        print(f"Test AP:          {test_ap:.4f}")


    # TABELLA FINALE
    # si stampa una tabella per confrontare le metriche

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

    # restituisce tutti i risultati per permettere al codice chiamante
    # di salvarli in un CSV, creare grafici o eseguire altre analisi.
    return results 


