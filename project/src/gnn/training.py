import torch
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

from .data import (
    prepare_edges,
    sample_message_passing_edges
)


def compute_loss(
    logits,
    labels
):

    return F.binary_cross_entropy_with_logits(
        logits,
        labels
    )


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