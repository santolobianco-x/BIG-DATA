import torch

from .data import prepare_edges


def score_edges_from_embeddings(
    z,
    decoder,
    edge_index,
    batch_size=50000
):
    """
    Calcola gli score degli archi a batch.
    """

    edge_index = edge_index.long()

    decoder.eval()

    scores = []

    with torch.no_grad():

        for start in range(
            0,
            edge_index.size(1),
            batch_size
        ):

            end = min(
                start + batch_size,
                edge_index.size(1)
            )

            batch_edges = edge_index[
                :,
                start:end
            ]

            logits = decoder(
                z,
                batch_edges
            )

            probabilities = torch.sigmoid(
                logits
            )

            scores.append(
                probabilities.cpu()
            )

    if len(scores) == 0:

        return torch.empty(
            0,
            dtype=torch.float
        )

    return torch.cat(
        scores
    )


def predict_edges(
    model,
    decoder,
    data,
    edge_index
):

    model.eval()
    decoder.eval()

    with torch.no_grad():

        z = model(
            data.x,
            data.edge_index
        )

    return score_edges_from_embeddings(
        z,
        decoder,
        edge_index
    )


def predict_query_candidates(
    model,
    decoder,
    data,
    query_positive_edges,
    query_negative_edges,
    node_to_idx,
    batch_size=50000
):
    """
    Calcola gli score per Hits@K.

    Gli embedding dei nodi vengono calcolati una sola volta.

    Tutti i negativi delle query vengono inoltre
    concatenati in un unico tensor e processati a batch.
    """

    model.eval()
    decoder.eval()

    with torch.no_grad():

        z = model(
            data.x,
            data.edge_index
        )

    positive_edge_index, _ = prepare_edges(
        query_positive_edges,
        [],
        node_to_idx
    )

    positive_edge_index = (
        positive_edge_index.to(
            data.x.device
        )
    )

    positive_scores = score_edges_from_embeddings(
        z,
        decoder,
        positive_edge_index,
        batch_size=batch_size
    )

    num_queries = len(
        query_negative_edges
    )

    if num_queries == 0:

        return (
            positive_scores.tolist(),
            []
        )

    negatives_per_query = len(
        query_negative_edges[0]
    )

    for negative_edges in query_negative_edges:

        if len(negative_edges) != negatives_per_query:

            raise ValueError(
                "Tutte le query devono avere "
                "lo stesso numero di negativi."
            )

    flat_negative_edges = [
        edge
        for query_edges in query_negative_edges
        for edge in query_edges
    ]

    flat_negative_edge_index, _ = prepare_edges(
        [],
        flat_negative_edges,
        node_to_idx
    )

    flat_negative_edge_index = (
        flat_negative_edge_index.to(
            data.x.device
        )
    )

    flat_negative_scores = (
        score_edges_from_embeddings(
            z,
            decoder,
            flat_negative_edge_index,
            batch_size=batch_size
        )
    )

    negative_scores = []

    for query_index in range(
        num_queries
    ):

        start = (
            query_index
            * negatives_per_query
        )

        end = (
            start
            + negatives_per_query
        )

        negative_scores.append(
            flat_negative_scores[
                start:end
            ].tolist()
        )

    return (
        positive_scores.tolist(),
        negative_scores
    )