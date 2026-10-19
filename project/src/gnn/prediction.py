
import torch

from .data import prepare_edges


def score_edges_from_embeddings(
    z,
    decoder,
    edge_index,
    batch_size=50000,
    return_probabilities=False,
):
    """
    Calcola gli score degli archi a batch.

    Per impostazione predefinita restituisce i logit,
    evitando la saturazione della sigmoid durante il ranking.

    Se return_probabilities=True, restituisce le probabilità
    ottenute applicando la sigmoid ai logit.
    """

    if batch_size <= 0:
        raise ValueError("batch_size deve essere maggiore di 0.")

    edge_index = edge_index.long()
    decoder.eval()

    scores = []

    with torch.no_grad():
        for start in range(0, edge_index.size(1), batch_size):
            end = min(start + batch_size, edge_index.size(1))
            batch_edges = edge_index[:, start:end]

            logits = decoder(z, batch_edges).reshape(-1)

            if return_probabilities:
                logits = torch.sigmoid(logits)

            scores.append(logits.cpu())

    if not scores:
        return torch.empty(0, dtype=z.dtype, device="cpu")

    return torch.cat(scores)


def predict_edges(
    model,
    decoder,
    data,
    edge_index,
    return_probabilities=False,
):
    """
    Calcola gli score degli archi usando il grafo di message passing
    presente in data.edge_index.
    """

    model.eval()
    decoder.eval()

    with torch.no_grad():
        z = model(data.x, data.edge_index)

    return score_edges_from_embeddings(
        z,
        decoder,
        edge_index,
        return_probabilities=return_probabilities,
    )


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
    Calcola gli score per Hits@K.

    Gli embedding dei nodi vengono calcolati una sola volta.
    I candidati negativi vengono concatenati e processati a batch.
    """

    model.eval()
    decoder.eval()

    with torch.no_grad():
        z = model(data.x, data.edge_index)

    positive_edge_index, _ = prepare_edges(
        query_positive_edges,
        [],
        node_to_idx,
    )

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
        raise ValueError("Ogni query deve avere almeno un negativo.")

    for negative_edges in query_negative_edges:
        if len(negative_edges) != negatives_per_query:
            raise ValueError(
                "Tutte le query devono avere lo stesso numero di negativi."
            )

    flat_negative_edges = [
        edge
        for query_edges in query_negative_edges
        for edge in query_edges
    ]

    flat_negative_edge_index, _ = prepare_edges(
        [],
        flat_negative_edges,
        node_to_idx,
    )

    flat_negative_edge_index = flat_negative_edge_index.to(data.x.device)

    flat_negative_scores = score_edges_from_embeddings(
        z,
        decoder,
        flat_negative_edge_index,
        batch_size=batch_size,
        return_probabilities=return_probabilities,
    )

    negative_scores = [
        flat_negative_scores[
            i * negatives_per_query:(i + 1) * negatives_per_query
        ].tolist()
        for i in range(num_queries)
    ]

    return positive_scores.tolist(), negative_scores
