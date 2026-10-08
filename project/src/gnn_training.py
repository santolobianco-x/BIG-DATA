import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# EDGE PREPARATION
# ============================================================

def prepare_edges(
    positive_edges,
    negative_edges,
    node_to_idx
):
    """
    Converte archi NetworkX in edge_index PyTorch Geometric.

    Gli archi positivi vengono assegnati label 1.
    Gli archi negativi vengono assegnati label 0.
    """

    edges = []
    labels = []

    for u, v in positive_edges:
        edges.append([
            node_to_idx[u],
            node_to_idx[v]
        ])
        labels.append(1.0)

    for u, v in negative_edges:
        edges.append([
            node_to_idx[u],
            node_to_idx[v]
        ])
        labels.append(0.0)

    if len(edges) == 0:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

        labels = torch.empty(
            (0,),
            dtype=torch.float
        )

        return edge_index, labels

    edge_index = torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()

    labels = torch.tensor(
        labels,
        dtype=torch.float
    )

    return edge_index, labels


# ============================================================
# MESSAGE PASSING EDGE SAMPLING
# ============================================================

def sample_message_passing_edges(
    edge_index,
    keep_ratio,
    seed
):
    """
    Seleziona casualmente una percentuale degli archi
    utilizzati per il message passing.
    """

    if not 0.0 <= keep_ratio <= 1.0:
        raise ValueError(
            "keep_ratio deve essere compreso tra 0 e 1."
        )

    if edge_index.numel() == 0:

        return torch.empty(
            (2, 0),
            dtype=edge_index.dtype,
            device=edge_index.device
        )

    if keep_ratio == 1.0:
        return edge_index

    if keep_ratio == 0.0:

        return torch.empty(
            (2, 0),
            dtype=edge_index.dtype,
            device=edge_index.device
        )

    edges = edge_index.t().tolist()

    undirected_edges = set()

    for u, v in edges:

        edge = tuple(sorted((u, v)))

        undirected_edges.add(edge)

    undirected_edges = list(
        undirected_edges
    )

    rng = random.Random(seed)

    rng.shuffle(
        undirected_edges
    )

    n_keep = int(
        len(undirected_edges)
        * keep_ratio
    )

    selected_edges = (
        undirected_edges[:n_keep]
    )

    directed_edges = []

    for u, v in selected_edges:

        directed_edges.append(
            [u, v]
        )

        directed_edges.append(
            [v, u]
        )

    if len(directed_edges) == 0:

        return torch.empty(
            (2, 0),
            dtype=edge_index.dtype,
            device=edge_index.device
        )

    return torch.tensor(
        directed_edges,
        dtype=edge_index.dtype,
        device=edge_index.device
    ).t().contiguous()


# ============================================================
# DECODER
# ============================================================

class MLPDecoder(nn.Module):

    def __init__(
        self,
        embedding_dim,
        hidden_dim
    ):
        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(
                embedding_dim * 2,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(
        self,
        z,
        edge_index
    ):

        edge_index = edge_index.long()

        src = edge_index[0]
        dst = edge_index[1]

        z_src = z[src]
        z_dst = z[dst]

        product = z_src * z_dst

        difference = torch.abs(
            z_src - z_dst
        )

        edge_features = torch.cat(
            [
                product,
                difference
            ],
            dim=1
        )

        return self.mlp(
            edge_features
        ).view(-1)


# ============================================================
# LOSS
# ============================================================

def compute_loss(
    logits,
    labels
):

    return F.binary_cross_entropy_with_logits(
        logits,
        labels
    )


# ============================================================
# TRAINING
# ============================================================

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
    seed=42
):

    train_edge_index, train_labels = prepare_edges(
        train_edges,
        train_negative_edges,
        node_to_idx
    )

    val_edge_index, val_labels = prepare_edges(
        val_edges,
        val_negative_edges,
        node_to_idx
    )

    train_edge_index = train_edge_index.to(
        data.x.device
    )

    train_labels = train_labels.to(
        data.x.device
    )

    val_edge_index = val_edge_index.to(
        data.x.device
    )

    val_labels = val_labels.to(
        data.x.device
    )

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

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()
        decoder.train()

        optimizer.zero_grad()

        message_passing_edge_index = (
            sample_message_passing_edges(
                data.edge_index,
                keep_ratio=keep_ratio,
                seed=seed + epoch
            )
        )

        message_passing_edge_index = (
            message_passing_edge_index.to(
                data.x.device
            )
        )

        z = model(
            data.x,
            message_passing_edge_index
        )

        train_logits = decoder(
            z,
            train_edge_index
        )

        loss = compute_loss(
            train_logits,
            train_labels
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()
        decoder.eval()

        with torch.no_grad():

            val_logits = decoder(
                z,
                val_edge_index
            )

            val_probabilities = torch.sigmoid(
                val_logits
            )

        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score
        )

        val_labels_cpu = (
            val_labels
            .detach()
            .cpu()
            .numpy()
        )

        val_probabilities_cpu = (
            val_probabilities
            .detach()
            .cpu()
            .numpy()
        )

        val_auc = roc_auc_score(
            val_labels_cpu,
            val_probabilities_cpu
        )

        val_ap = average_precision_score(
            val_labels_cpu,
            val_probabilities_cpu
        )

        history["loss"].append(
            loss.item()
        )

        history["val_auc"].append(
            val_auc
        )

        history["val_ap"].append(
            val_ap
        )

        if val_auc > best_val_auc:

            best_val_auc = val_auc
            best_val_ap = val_ap
            best_epoch = epoch

            best_model_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

            best_decoder_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in decoder.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):

            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"Val AP: {val_ap:.4f}"
            )

    # --------------------------------------------------------
    # RESTORE BEST CHECKPOINT
    # --------------------------------------------------------

    if best_model_state is not None:

        model.load_state_dict(
            best_model_state
        )

    if best_decoder_state is not None:

        decoder.load_state_dict(
            best_decoder_state
        )

    history["best_epoch"] = best_epoch
    history["best_val_auc"] = best_val_auc
    history["best_val_ap"] = best_val_ap

    print()
    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation AUC: "
        f"{best_val_auc:.4f}"
    )

    print(
        f"Validation AP at best epoch: "
        f"{best_val_ap:.4f}"
    )

    return history


# ============================================================
# SCORE EDGES
# ============================================================

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


# ============================================================
# STANDARD EDGE PREDICTION
# ============================================================

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


# ============================================================
# QUERY-BASED PREDICTION
# ============================================================

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
    concatenati in un unico tensor e processati a batch,
    evitando 1000 chiamate separate al decoder.
    """

    model.eval()
    decoder.eval()

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    with torch.no_grad():

        z = model(
            data.x,
            data.edge_index
        )

    # --------------------------------------------------------
    # POSITIVES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # NEGATIVES
    # --------------------------------------------------------

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

    # Controlliamo che tutte le query abbiano
    # lo stesso numero di candidati.
    for negative_edges in query_negative_edges:

        if len(negative_edges) != negatives_per_query:

            raise ValueError(
                "Tutte le query devono avere "
                "lo stesso numero di negativi."
            )

    flat_negative_edges = [

        edge

        for query_edges
        in query_negative_edges

        for edge
        in query_edges
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

    # --------------------------------------------------------
    # RESTORE QUERY STRUCTURE
    # --------------------------------------------------------

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