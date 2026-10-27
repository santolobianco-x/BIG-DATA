"""
MODULO PER L'ORCHESTRAZIONE DEGLI ESPERIMENTI CON GNN

--> gestisce la preparazione dei dati in formato PYG, il controllo rigoroso
    dell'anti-leakage, studio di ablazione e confronto tra diversi decoder.
"""
import numpy as np
import torch

from src.gnn import (
    networkx_to_pyg,
    GraphSAGE,
    MLPDecoder,
    DotProductDecoder,
    train_gnn,
    predict_edges,
    predict_query_candidates,
    prepare_edges
)

from src.evaluation import hits_at_k, evaluate_scores


from config import (
    DEVICE,
    GNN_SEEDS,
    RESULTS_DIR,
    DATASET_NAME,
    HIDDEN_CHANNELS,
    OUT_CHANNELS,
    DECODER_HIDDEN_DIM,
    LEARNING_RATE,
    EPOCHS,
    ABLATION_EPOCHS,
    ABLATION_SEED,
    ABLATION_LEARNING_RATE,
    HITS_K_VALUES
)

from src.utils.sanity_checks import verify_feature_source
from experiments.run_ablation import run_feature_ablation


from experiments.export_results import export_rows



EVALUATION_METRICS = [
    "val_auc", "val_ap", "test_auc", "test_ap"
] + [f"hits@{k}" for k in HITS_K_VALUES]


def prepare_gnn_data(message_passing_graph, print_section):
    """
    Traduce il grafo in tensori per la GNN. Il grafo in input deve
    essere il message_passing_graph, privato degli archi di supervisione
    e di test, per costringere l'Encoder a imparare le rappresentazioni 
    osservando solo le connessioni ammesse.
    """

    print_section("GNN DATA")

    data, _ = networkx_to_pyg(
        message_passing_graph,
        feature_mode="full",
    )

    data = data.to(DEVICE)

    print(f"Nodes: {data.num_nodes}")
    print(f"Node features: {data.x.shape[1]}")
    print(f"Message-passing edges: {data.edge_index.shape[1]}")

    print_section("FEATURE SOURCE CHECK")

    return data


def run_gnn_multiseed(
    message_passing_graph,
    supervised_train_edges,
    supervised_train_negative_edges,
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
    learning_rate=0.001,
    decoder_type="mlp",
    device="cpu"
):
    """
    Esegue l'addestramento e la valutazione della GNN 
    ripetutamente su seed diversi.
    Si aggregano run diverse, con inizializzazione casuale dei pesi,
    per restituire media e deviazione standard e garantire robustezza
    statistica.
    """

    if decoder_type not in ("mlp", "dot_product"):
        raise ValueError(
            "decoder_type deve essere 'mlp' oppure 'dot_product'."
        )

    results = []

    # preparazione del grafo base
    data, node_to_idx = networkx_to_pyg(message_passing_graph,feature_mode="full")
    data = data.to(device)
    
    # itera per i seed che sono stati indicati
    for seed in seeds:

        print()
        print("=" * 70)
        print(f"GRAPHSAGE + {decoder_type.upper()} - SEED {seed}")
        print("=" * 70)

        
        # si impostano i seed per rendere riproducibili le operazioni casuali di PyTorch e NumPy
        torch.manual_seed(seed)
        np.random.seed(seed)

        # inizializzazione della rete neurale
        model = GraphSAGE(
            in_channels=data.x.shape[1],
            hidden_channels=hidden_channels,
            out_channels=out_channels
        ).to(device)


        # selezione dell'architettura del decoder
        if decoder_type == "mlp":
            decoder = MLPDecoder(
                embedding_dim=out_channels,
                hidden_dim=decoder_hidden_dim
            ).to(device)

        else:
            decoder = DotProductDecoder().to(device)


        # si crea l'ottimizzatore Adam
        # l'ottimizzatore deve aggiornare sia i pesi dell'encoder
        # e sia i pesi del decoder --> quanto questo contiene pesi addestrabili 
        optimizer = torch.optim.Adam(
            list(model.parameters())
            + list(decoder.parameters()),
            lr=learning_rate
        )

        # apprendimento da parte della rete e conservazione dei risultati
        # del modello in un dizionario
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


        # si preparano separatamente gli archi positivi e negativi di test
        test_pos_edge_index, _ = prepare_edges(test_edges, [], node_to_idx)
        test_neg_edge_index, _ = prepare_edges([], test_negative_edges, node_to_idx)

        #si spostano i tensori sul dispositivo del modello
        test_pos_edge_index = test_pos_edge_index.to(device)
        test_neg_edge_index = test_neg_edge_index.to(device)

        # si uniscono gli archi positivi e gli archi negativi
        test_edge_index = torch.cat([test_pos_edge_index,test_neg_edge_index],dim=1)

        # il modello calcola un punteggio per ogni coppia di nodi
        # --> punteggi utilizzati per valutare quanto bene distingue
        #     archi esistenti da archi non esistenti
        test_scores = predict_edges(model, decoder, data, test_edge_index)
        
        n_positive = len(test_edges)
    

        # confronto punteggi positivi e negativi, restituendo AUC e AP
        test_metrics = evaluate_scores(pos_scores=test_scores[:n_positive],neg_scores=test_scores[n_positive:])
        
        test_auc = test_metrics["auc"]
        test_ap = test_metrics["ap"]

        # si calcola la metrica hits@K per le coppie positive e negative
        # associate alle query. La valutazione viene effettuata sulle coppie
        # di test
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

        hits_results = {
            f"hits@{k}": hits_at_k(query_positive_scores, query_negative_scores, k)
            for k in HITS_K_VALUES
        }


        # salvataggio delle informazioni principali relative al 
        # modello addestrato con il seed specificato
        result = {
            "decoder": decoder_type,
            "seed": seed,
            "best_epoch": history["best_epoch"],
            "epochs_trained": history["epochs_trained"],
            "val_auc": history["best_val_auc"],
            "val_ap": history["best_val_ap"],
            "test_auc": test_auc,
            "test_ap": test_ap,
        }


        result.update(hits_results)# si aggiungono al dizionario le metriche Hits@K

        results.append(result) # si inserisce il risultato nella lista complessiva

        # STAMPA DEI RISULTATI DEL SINGOLO SEED
        print()
        print(f"Seed:       {seed}")
        print(f"Best epoch: {history['best_epoch']}")
        print(f"Val AUC:    {history['best_val_auc']:.4f}")
        print(f"Val AP:     {history['best_val_ap']:.4f}")
        print(f"Test AUC:   {test_auc:.4f}")
        print(f"Test AP:    {test_ap:.4f}")

        # stampa delle metriche hits@K
        for k in HITS_K_VALUES:
            print(f"Hits@{k}:    {hits_results[f'hits@{k}']:.4f}")


    # se non è stato eseguito nemmeno un seed, non si può calcolare media 
    # e deviazioni standard
    if not results:
        raise ValueError("La lista dei seed non può essere vuota.")

    
    print()
    print("=" * 70)
    print("MULTI-SEED SUMMARY")
    print("=" * 70)

    # dizionario in cui si salva media e deviazione standard di ciascuna metrica
    summary = {}

    for metric in EVALUATION_METRICS:

        values = np.array([result[metric] for result in results])

        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }

        print(f"{metric:<10}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # per le metriche principali si stampa anche il minimo e il massimo 
    # ottenuto tra i seed
    for metric in ["test_auc", "test_ap"] + [f"hits@{k}" for k in HITS_K_VALUES]:

        values = [
            result[metric]
            for result in results
        ]

        print()
        print(f"{metric} range:")
        print(f"  min = {np.min(values):.4f}")
        print(f"  max = {np.max(values):.4f}")

    return results # restituisce la lista dei risultati 




def run_gnn_experiments(splits, negatives, queries, 
                        original_graph, data,print_section,
):
    """
    Coordina gli esperimenti GNN del progetto, esegue:
    - controlli sulle feature;
    - feature ablation
    - confronto tra decoder.
    """

    # si estraggono dal dizionario degli split:
    # - grafo usato per message passing;
    # - archi supervisionati.

    message_passing_graph = splits["message_passing_graph"]

    supervised_train_edges = splits["supervised_train_edges"]
    
    supervised_train_negative_edges = negatives["supervised_train_negative_edges"]

    # si estraggono gli archi di validation e test
    val_edges = splits["val_edges"]
    test_edges = splits["test_edges"]

    # si controlla che le feature siano coerenti con il grafo ammesso per il training
    # in particolare se le feature siano state estratte esclusivamente dal training graph
    if not verify_feature_source(
        original_graph,
        message_passing_graph,
        data,
    ):
        raise RuntimeError("Feature GNN non coerenti con il grafo di training.")



    # si esegue lo studio di ablation, provando le configurazioni:
    # full, degree, clustering e costant.
    # per ciascuna configurazione addestra e valuta una GNN.


    feature_ablation_results = run_feature_ablation(
        message_passing_graph=message_passing_graph,
        supervised_train_edges=supervised_train_edges,
        supervised_train_negative_edges=supervised_train_negative_edges,
        val_edges=val_edges,
        val_negative_edges=negatives["val_negative_edges"],
        test_edges=test_edges,
        test_negative_edges=negatives["test_negative_edges"],
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
        decoder_hidden_dim=DECODER_HIDDEN_DIM,
        learning_rate=ABLATION_LEARNING_RATE,
        epochs=ABLATION_EPOCHS,
        seed=ABLATION_SEED,
        device=DEVICE,
    )


    # si preprara una lista di righe da esportare nel CSV, dove ogni riga
    # corrisponde ad una modalità di feature
    feature_ablation_rows = []

    # il dizionario restituito dall'ablation contiene una voce per ogni modalità
    # e le relative metriche
    for mode, metrics in feature_ablation_results.items():
        feature_ablation_rows.append({
            "feature_mode": mode,
            "input_features": metrics["input_features"],
            "best_epoch": metrics["best_epoch"],
            "epochs_trained": metrics["epochs_trained"],
            "val_auc": metrics["val_auc"],
            "val_ap": metrics["val_ap"],
            "test_auc": metrics["test_auc"],
            "test_ap": metrics["test_ap"],
        })

    # si esportano i risultati dell'ablation nel file CSV
    export_rows(
        feature_ablation_rows,
        RESULTS_DIR / f"{DATASET_NAME}_feature_ablation_results.csv",
    )

    print_section("GNN DECODER COMPARISON")

    query_positive_edges, query_negative_edges = queries

    # esegue gli esperimenti per entrambi i decoder
    gnn_results = {}
    for decoder_type in ("mlp", "dot_product"):
        print_section(f"GRAPHSAGE + {decoder_type.upper()}")

        # per ciascun decoder si chiama la funzione multi-seed
        gnn_results[decoder_type] = run_gnn_multiseed(
            message_passing_graph=message_passing_graph,
            supervised_train_edges=supervised_train_edges,
            supervised_train_negative_edges=supervised_train_negative_edges,
            val_edges=val_edges,
            val_negative_edges=negatives["val_negative_edges"],
            test_edges=test_edges,
            test_negative_edges=negatives["test_negative_edges"],
            query_positive_edges=query_positive_edges,
            query_negative_edges=query_negative_edges,
            seeds=GNN_SEEDS,
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=OUT_CHANNELS,
            decoder_hidden_dim=DECODER_HIDDEN_DIM,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            decoder_type=decoder_type,
            device=DEVICE,
        )

    # si restituisce entrambi i gruppi di risultati
    return {
        "feature_ablation": feature_ablation_results,
        "decoder_comparison": gnn_results,
    }



def save_gnn_results(gnn_results, print_section):
    """
    Salva i risultati del confronto decoder, producendo:
    - CSV con i risultati dettagliati per ogni seed
    - CSV con media e deviazione standard
    """
    print_section("GNN FINAL RESULTS")    

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


    # si unisce in un'unica lista i risultati di tutti i decoder
    all_gnn_results = [
        result
        for decoder_results in gnn_results.values()
        for result in decoder_results
    ]

    # se non ci sono risultati non ha senso creare CSV
    if not all_gnn_results:
        raise RuntimeError("Nessun risultato GNN disponibile.")

    detailed_path = RESULTS_DIR / f"{DATASET_NAME}_gnn_decoder_results.csv"

    export_rows(all_gnn_results, detailed_path)


    # lista contenente tutte le righe del CSV riepilogativo
    summary_rows = []
    print()


    # si prepara l'intestazione della tabella da stampare
    header = f"{'Model':<25}{'Test AUC':>23}{'Test AP':>23}"
    header += "".join([f"Hits@{k}".rjust(23) for k in HITS_K_VALUES])


    print(header)
    print("-" * len(header))



    # si analizzano separatamente i risultati di ogni decoder
    for decoder_type, decoder_results in gnn_results.items():

        # si saltano eventuali decoder senza risultati
        if not decoder_results:
            continue

        
        # per ogni metrica si raccolgono i valori ottenuti dai diversi seed  
        values = {
            metric: np.asarray(
                [result[metric] for result in decoder_results],
                dtype=float,
            )
            for metric in EVALUATION_METRICS
        }

        label = ("GraphSAGE + MLP" if decoder_type == "mlp" 
                 else "GraphSAGE + Dot Product")

        # si selezionano le metrcihe che si vuole mostrare nella tabella
        displayed_metrics = (
            "test_auc",
            "test_ap",
        ) + tuple(f"hits@{k}" for k in HITS_K_VALUES)

        # si stampa media e deviazione standard per ciascuna delle metriche selezionate
        print(
            f"{label:<25}"
            + "".join(
                f"{np.mean(values[m]):.4f} ± {np.std(values[m]):.4f}".rjust(23)
                for m in displayed_metrics
            )
        )

        # si crea una riga riepilogativa per il CSV
        row = {"decoder": decoder_type, "num_seeds": len(decoder_results)}

        # per ogni metrica si salva la media e la deviazione standard
        for metric in EVALUATION_METRICS:
            row[f"{metric}_mean"] = float(np.mean(values[metric]))
            row[f"{metric}_std"] = float(np.std(values[metric]))
        
        summary_rows.append(row)

    # si definisce il percorso del CSV riepilogativo
    summary_path = RESULTS_DIR / f"{DATASET_NAME}_gnn_decoder_summary.csv"

    # si esportano le statistiche 
    export_rows( summary_rows, summary_path)