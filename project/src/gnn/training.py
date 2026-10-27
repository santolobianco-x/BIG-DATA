"""
MODULO DEDICATO ALL'ADDESTRAMENTO DELLA GNN

--> implementazione del ciclo di training con ottimizzazione attraverso 
    discesa del gradiente, regolarizzazione mediante DropEdge ed Early Stopping
"""

import torch
import torch.nn.functional as F

from src.evaluation import evaluate_scores

from .data import (
    prepare_edges,
    sample_message_passing_edges
)


def compute_loss(logits, labels):
    """
    Calcola l'errore della rete attraverso Binary Cross Entropy(with Logits).
    Unisce la Sigmoide seguita da BCE(Binary Cross Entropy) evitando 
    instabilità numeriche se fosse fattoin modo separato
    """
    
    return F.binary_cross_entropy_with_logits(logits, labels)


def train_gnn(
    model,
    decoder,
    data,
    train_edges,
    train_negative_edges,
    val_edges,
    val_negative_edges,
    node_to_idx,
    optimizer,
    epochs,
    keep_ratio=1.0,
    seed=42,
    patience=50
):

    # preparazione dei tensori e caricamento sulla memoria della GPU
    train_edge_index, train_labels = prepare_edges(train_edges, train_negative_edges,node_to_idx)

    val_edge_index, val_labels = prepare_edges(val_edges, val_negative_edges, node_to_idx)

    
    train_edge_index = train_edge_index.to(data.x.device)
    train_labels = train_labels.to(data.x.device)
    val_edge_index = val_edge_index.to(data.x.device)
    val_labels = val_labels.to(data.x.device)

    # variabili di monitoraggio
    history = {
        "loss": [],
        "val_auc": [],
        "val_ap": [],
        "best_epoch": None,
        "best_val_auc": None,
        "best_val_ap": None
    }


    best_val_auc = -float("inf")
    best_val_ap = None
    best_epoch = None
    best_model_state = None
    best_decoder_state = None
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch in range(1, epochs + 1):
        #modalità di addestramento
        model.train()
        decoder.train()

        # resetta i gradienti calcolati nell'epoca precedente
        optimizer.zero_grad()

        message_passing_edge_index = (
            sample_message_passing_edges(
                data.edge_index,
                keep_ratio=keep_ratio,
                seed=seed + epoch
            )
        ) # sottocampiona gli archi strutturali, cambiando a ogni epoca il seed

        message_passing_edge_index = message_passing_edge_index.to(data.x.device)

        # calcola gli embedding dei nodi
        z = model(data.x, message_passing_edge_index)

        # calcola gli score per gli archi di supervisione
        train_logits = decoder(z, train_edge_index)

        # calcola l'errore prodotto dalla rete
        loss = compute_loss(train_logits,train_labels)

        # calcola la derivata per aggiornare i pesi
        loss.backward()

        # aggiorna i pesi della rete
        optimizer.step()

        # disabilità aggiornamenti dei pesi
        model.eval()
        decoder.eval()

        with torch.no_grad():
            # in validazione si utilizza lo scheletro completo per avere gli embedding più accurati
            z_val = model(data.x,data.edge_index)

            val_scores = decoder(z_val, val_edge_index)

        # si separano gli score assegnati agli archi veri da quelli assegnati agli archi falsi
        pos_mask = val_labels == 1.0
        neg_mask = val_labels == 0.0

        val_metrics = evaluate_scores(
            pos_scores=val_scores[pos_mask],
            neg_scores=val_scores[neg_mask]
        )

        val_auc = val_metrics["auc"]
        val_ap = val_metrics["ap"]

        history["loss"].append(loss.item())
        history["val_auc"].append(val_auc)
        history["val_ap"].append(val_ap)

        if val_auc > best_val_auc:
            # si ha un nuovo record e si aggiornano le prestazioni
            best_val_auc = val_auc
            best_val_ap = val_ap
            best_epoch = epoch
            epochs_without_improvement = 0

            # si copia lo stato esatto dei pesi del modello
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            best_decoder_state = {
                key: value.detach().cpu().clone()
                for key, value
                in decoder.state_dict().items()
            }

        # se non c'è nessun miglioramento si aggiorna il numero di epoche senza miglioramenti
        else:
            epochs_without_improvement += 1

        epochs_trained = epoch

        # stampa a schermo i progressi ogni 10 epoche
        if (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"Val AP: {val_ap:.4f}"
            )

        # stampa in caso di early stopping
        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}: "
                f"validation AUC did not improve for "
                f"{patience} consecutive epochs."
            )
            break

    # alla fine del ciclo, ripristina la rete ai pesi dell'epoca migliore
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if best_decoder_state is not None:
        decoder.load_state_dict(best_decoder_state)

    history["best_epoch"] = best_epoch
    history["best_val_auc"] = best_val_auc
    history["best_val_ap"] = best_val_ap
    history["epochs_trained"] = epochs_trained

    print()
    print(f"Best epoch: {best_epoch}")

    print(f"Best validation AUC: {best_val_auc:.4f}")

    print(f"Validation AP at best epoch: {best_val_ap:.4f}")

    return history # ritorna lo storico dell'addestramento