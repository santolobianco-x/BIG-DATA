"""
MODULO PER L'INFERENENZA DELLA GNN.

--> mette a disposizione le funzioni per far produrre al modello 
    le previsioni sui link futuri.
    L'elaborazione è a blocchi per evitare errori di memoria.
"""
import torch

from .data import prepare_edges


def score_edges_from_embeddings(z,decoder, edge_index,batch_size=50000,return_probabilities=False):
    """
    Calcola gli score degli archi a batch. Prende gli embedding calcolati e usa
    il decoder per valutare enormi quantità di archi.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    edge_index = edge_index.long()

    # Decoder in modalità valutazione
    decoder.eval()

    scores = []

    with torch.no_grad(): # disabilita il calcolo dei gradienti(RAM risparmiata)


        for start in range(0, edge_index.size(1), batch_size): # itera per batch
            end = min(start + batch_size, edge_index.size(1))
            batch_edges = edge_index[:, start:end]

            # passa gli archi al decoder
            logits = decoder(z, batch_edges).reshape(-1)

            # ritorna i logit tra 0 e 1 utilizzando la funzione Sigmoide
            if return_probabilities:
                logits = torch.sigmoid(logits)

            # gli score vengono spostati nella CPU
            scores.append(logits.cpu())

    if not scores:
        return torch.empty(0, dtype=z.dtype, device="cpu")

    return torch.cat(scores) # ricompone tutti i pezzetti in un unico tensore


def predict_edges(model, decoder, data, edge_index,return_probabilities=False):
    """
    Pipeline completa di predizione per gli archi globali. Esegue l'encoder
    per calcolare l'embedding dei nodi e poi utilizza il Decoder per predire una
    lista di archi.
    """

    model.eval()
    decoder.eval()

    with torch.no_grad():
        # costruisce gli embedding guardando solo il training graph
        z = model(data.x, data.edge_index)

    # il decoder ritornerà lo score agli archi passati
    return score_edges_from_embeddings(z, decoder,edge_index, return_probabilities=return_probabilities)


def predict_query_candidates(
    model,
    decoder,
    data,
    query_positive_edges,
    query_negative_edges,
    node_to_idx,
    batch_size=50000,
    return_probabilities=False,
):
    """
    Calcola gli score per Hits@K. Si valutano milioni di archi falsi
    per questo motivo:
    - calcola l'embedding della GNN una sola volta per tutti i nodi
    - appiattisce le liste di distrattotri in un'unica lista globale
    - valuta i distrattori a blocchi
    - restituisce i risultati restituendoli al formato query per query
    """

    model.eval()
    decoder.eval()

    with torch.no_grad():
        z = model(data.x, data.edge_index)

    positive_edge_index, _ = prepare_edges(query_positive_edges, [],node_to_idx,)

    positive_edge_index = positive_edge_index.to(data.x.device)

    positive_scores = score_edges_from_embeddings(
        z,
        decoder,
        positive_edge_index,
        batch_size=batch_size,
        return_probabilities=return_probabilities,
    )

    num_queries = len(query_negative_edges)

    if num_queries == 0:
        return positive_scores.tolist(), []

    negatives_per_query = len(query_negative_edges[0])

    if negatives_per_query == 0:
        raise ValueError("Each query must have at least one negative.")

    for negative_edges in query_negative_edges:
        if len(negative_edges) != negatives_per_query:
            raise ValueError("All queries must have the same number of negatives.")

    flat_negative_edges = [
        edge
        for query_edges in query_negative_edges
        for edge in query_edges
    ] # si mette in un'unica lista le liste di liste

    flat_negative_edge_index, _ = prepare_edges([], flat_negative_edges,node_to_idx,)

    flat_negative_edge_index = flat_negative_edge_index.to(data.x.device)

    flat_negative_scores = score_edges_from_embeddings(
        z,
        decoder,
        flat_negative_edge_index,
        batch_size=batch_size,
        return_probabilities=return_probabilities,
    ) # restituisce i punteggi degli archi negativi

    negative_scores = [
        flat_negative_scores[
            i * negatives_per_query:(i + 1) * negatives_per_query
        ].tolist()
        for i in range(num_queries)
    ] # rimette nelle liste associate ai relativi positive edge gli score degli archi negativi

    return positive_scores.tolist(), negative_scores # restituisce le liste pronte
