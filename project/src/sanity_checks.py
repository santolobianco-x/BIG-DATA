import torch


# ============================================================
# EDGE UTILITIES
# ============================================================

def canonical_edge(u, v):
    """
    Restituisce un arco non orientato in forma canonica.
    """

    return tuple(sorted((u, v)))


def canonical_edges(edges):
    """
    Converte una lista di archi nella rappresentazione canonica.
    """

    return {
        canonical_edge(u, v)
        for u, v in edges
    }


# ============================================================
# POSITIVE EDGE SANITY CHECK
# ============================================================

def verify_positive_split(
    G,
    train_graph,
    val_edges,
    test_edges
):
    """
    Verifica che gli archi positivi siano:

    - presenti nel grafo originale;
    - mutuamente disgiunti;
    - correttamente distribuiti tra train, validation e test.
    """

    print()
    print("=" * 60)
    print("POSITIVE EDGE SANITY CHECK")
    print("=" * 60)

    original_edges = canonical_edges(
        G.edges()
    )

    train_edges = canonical_edges(
        train_graph.edges()
    )

    val_edges_set = canonical_edges(
        val_edges
    )

    test_edges_set = canonical_edges(
        test_edges
    )

    print(
        f"Original edges : {len(original_edges)}"
    )

    print(
        f"Train edges    : {len(train_edges)}"
    )

    print(
        f"Validation     : {len(val_edges_set)}"
    )

    print(
        f"Test           : {len(test_edges_set)}"
    )

    # --------------------------------------------------------
    # Membership
    # --------------------------------------------------------

    train_valid = train_edges <= original_edges
    val_valid = val_edges_set <= original_edges
    test_valid = test_edges_set <= original_edges

    print()
    print("Membership:")

    print(
        f"Train ⊆ Original: {train_valid}"
    )

    print(
        f"Validation ⊆ Original: {val_valid}"
    )

    print(
        f"Test ⊆ Original: {test_valid}"
    )

    # --------------------------------------------------------
    # Disjointness
    # --------------------------------------------------------

    train_val = train_edges & val_edges_set
    train_test = train_edges & test_edges_set
    val_test = val_edges_set & test_edges_set

    print()
    print("Disjointness:")

    print(
        f"Train ∩ Validation : {len(train_val)}"
    )

    print(
        f"Train ∩ Test       : {len(train_test)}"
    )

    print(
        f"Validation ∩ Test  : {len(val_test)}"
    )

    valid = (
        train_valid
        and val_valid
        and test_valid
        and len(train_val) == 0
        and len(train_test) == 0
        and len(val_test) == 0
    )

    print()

    if valid:
        print("✓ Positive split valido")
    else:
        print("✗ Positive split NON valido")

    return valid


# ============================================================
# NEGATIVE EDGE SANITY CHECK
# ============================================================

def verify_negative_edges(
    G,
    train_graph,
    train_negative_edges,
    val_negative_edges,
    test_negative_edges,
    val_edges,
    test_edges
):
    """
    Verifica che gli archi negativi:

    - non appartengano al grafo originale;
    - siano disgiunti tra loro;
    - non coincidano con gli archi positivi.
    """

    print()
    print("=" * 60)
    print("NEGATIVE EDGE SANITY CHECK")
    print("=" * 60)

    original_edges = canonical_edges(
        G.edges()
    )

    train_positive = canonical_edges(
        train_graph.edges()
    )

    val_positive = canonical_edges(
        val_edges
    )

    test_positive = canonical_edges(
        test_edges
    )

    train_negative = canonical_edges(
        train_negative_edges
    )

    val_negative = canonical_edges(
        val_negative_edges
    )

    test_negative = canonical_edges(
        test_negative_edges
    )

    print(
        f"Train negatives : {len(train_negative)}"
    )

    print(
        f"Validation negatives : {len(val_negative)}"
    )

    print(
        f"Test negatives : {len(test_negative)}"
    )

    # --------------------------------------------------------
    # Negatives vs original graph
    # --------------------------------------------------------

    train_original = (
        train_negative & original_edges
    )

    val_original = (
        val_negative & original_edges
    )

    test_original = (
        test_negative & original_edges
    )

    print()
    print("Negatives ∩ Original:")

    print(
        f"Train negatives ∩ Original : "
        f"{len(train_original)}"
    )

    print(
        f"Validation negatives ∩ Original : "
        f"{len(val_original)}"
    )

    print(
        f"Test negatives ∩ Original : "
        f"{len(test_original)}"
    )

    # --------------------------------------------------------
    # Negative sets disjointness
    # --------------------------------------------------------

    train_val = (
        train_negative & val_negative
    )

    train_test = (
        train_negative & test_negative
    )

    val_test = (
        val_negative & test_negative
    )

    print()
    print("Negative sets disjointness:")

    print(
        f"Train ∩ Validation : {len(train_val)}"
    )

    print(
        f"Train ∩ Test       : {len(train_test)}"
    )

    print(
        f"Validation ∩ Test  : {len(val_test)}"
    )

    # --------------------------------------------------------
    # Negative vs positive
    # --------------------------------------------------------

    train_neg_train_pos = (
        train_negative & train_positive
    )

    train_neg_val_pos = (
        train_negative & val_positive
    )

    train_neg_test_pos = (
        train_negative & test_positive
    )

    val_neg_val_pos = (
        val_negative & val_positive
    )

    test_neg_test_pos = (
        test_negative & test_positive
    )

    print()
    print("Negative vs positive sets:")

    print(
        f"Train negatives ∩ Train positives : "
        f"{len(train_neg_train_pos)}"
    )

    print(
        f"Train negatives ∩ Validation positives : "
        f"{len(train_neg_val_pos)}"
    )

    print(
        f"Train negatives ∩ Test positives : "
        f"{len(train_neg_test_pos)}"
    )

    print(
        f"Validation negatives ∩ Validation positives : "
        f"{len(val_neg_val_pos)}"
    )

    print(
        f"Test negatives ∩ Test positives : "
        f"{len(test_neg_test_pos)}"
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    valid = (
        len(train_original) == 0
        and len(val_original) == 0
        and len(test_original) == 0

        and len(train_val) == 0
        and len(train_test) == 0
        and len(val_test) == 0

        and len(train_neg_train_pos) == 0
        and len(train_neg_val_pos) == 0
        and len(train_neg_test_pos) == 0
        and len(val_neg_val_pos) == 0
        and len(test_neg_test_pos) == 0
    )

    print()

    if valid:
        print("✓ Negative sampling valido")
    else:
        print("✗ Negative sampling NON valido")

    return valid


# ============================================================
# ISOLATED NODE ANALYSIS
# ============================================================

def analyze_isolated_nodes(
    train_graph,
    val_edges,
    test_edges
):
    """
    Analizza quanti nodi diventano isolati nel training graph
    a causa dell'edge split e quanti archi di validation/test
    coinvolgono tali nodi.
    """

    print()
    print("=" * 60)
    print("ISOLATED NODE ANALYSIS")
    print("=" * 60)

    isolated_nodes = {
        node
        for node, degree
        in train_graph.degree()
        if degree == 0
    }

    print(
        f"Isolated nodes in train : "
        f"{len(isolated_nodes)}"
    )

    val_isolated = sum(
        1
        for u, v in val_edges
        if (
            u in isolated_nodes
            or v in isolated_nodes
        )
    )

    test_isolated = sum(
        1
        for u, v in test_edges
        if (
            u in isolated_nodes
            or v in isolated_nodes
        )
    )

    print(
        f"Validation edges with isolated endpoint : "
        f"{val_isolated} / {len(val_edges)} "
        f"({val_isolated / len(val_edges) * 100:.4f}%)"
    )

    print(
        f"Test edges with isolated endpoint : "
        f"{test_isolated} / {len(test_edges)} "
        f"({test_isolated / len(test_edges) * 100:.4f}%)"
    )

    return {
        "isolated_nodes": len(isolated_nodes),
        "validation_isolated_edges": val_isolated,
        "test_isolated_edges": test_isolated
    }


# ============================================================
# FEATURE COMPUTATION
# ============================================================

def compute_expected_features(G):
    """
    Calcola manualmente le feature utilizzate dal modello:

    1. grado normalizzato;
    2. clustering coefficient.
    """

    nodes = list(G.nodes())

    degrees = [
        G.degree(node)
        for node in nodes
    ]

    max_degree = max(degrees)

    clustering = {
        node: value
        for node, value
        in __import__("networkx").clustering(G).items()
    }

    features = torch.tensor(
        [
            [
                degree / max_degree,
                clustering[node]
            ]
            for node, degree
            in zip(nodes, degrees)
        ],
        dtype=torch.float
    )

    return features


# ============================================================
# FEATURE SOURCE CHECK
# ============================================================

def verify_feature_source(
    G,
    train_graph,
    data
):
    """
    Verifica che le feature utilizzate dalla GNN siano
    calcolate dal training graph e non dal grafo completo.

    Questo serve a escludere leakage informativo.
    """

    print()
    print("=" * 60)
    print("FEATURE SOURCE CHECK")
    print("=" * 60)

    original_features = compute_expected_features(
        G
    )

    train_features = compute_expected_features(
        train_graph
    )

    actual_features = (
        data.x.cpu()
    )

    max_difference_original = torch.max(
        torch.abs(
            actual_features
            -
            original_features
        )
    ).item()

    max_difference_train = torch.max(
        torch.abs(
            actual_features
            -
            train_features
        )
    ).item()

    matches_original = torch.allclose(
        actual_features,
        original_features
    )

    matches_train = torch.allclose(
        actual_features,
        train_features
    )

    print(
        f"Matches original graph: "
        f"{matches_original}"
    )

    print(
        f"Matches train graph: "
        f"{matches_train}"
    )

    print(
        f"Max difference vs original: "
        f"{max_difference_original:.8f}"
    )

    print(
        f"Max difference vs train: "
        f"{max_difference_train:.8f}"
    )

    print()

    if matches_train:
        print(
            "✓ Le feature corrispondono "
            "al training graph."
        )
    else:
        print(
            "✗ Le feature NON corrispondono "
            "al training graph."
        )

    return matches_train


# ============================================================
# ORIENTATION SENSITIVITY
# ============================================================

def evaluate_orientation_sensitivity(
    model,
    decoder,
    data,
    node_to_idx,
    positive_edges,
    negative_edges
):
    """
    Verifica se il decoder produce lo stesso punteggio
    per (u, v) e (v, u).

    Poiché il grafo è non orientato, il risultato dovrebbe
    essere invariato rispetto all'ordine dei due nodi.
    """

    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score
    )

    # --------------------------------------------------------
    # Convert NetworkX edges -> PyTorch edge_index
    # --------------------------------------------------------

    def convert_edges(edges):

        if len(edges) == 0:
            return torch.empty(
                (2, 0),
                dtype=torch.long
            )

        return torch.tensor(
            [
                [
                    node_to_idx[u],
                    node_to_idx[v]
                ]
                for u, v in edges
            ],
            dtype=torch.long
        ).t().contiguous()

    positive_index = convert_edges(
        positive_edges
    )

    negative_index = convert_edges(
        negative_edges
    )

    # --------------------------------------------------------
    # Reverse orientation
    # --------------------------------------------------------

    positive_reversed = (
        positive_index.flip(0)
    )

    negative_reversed = (
        negative_index.flip(0)
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = data.x.device

    positive_index = (
        positive_index.to(device)
    )

    negative_index = (
        negative_index.to(device)
    )

    positive_reversed = (
        positive_reversed.to(device)
    )

    negative_reversed = (
        negative_reversed.to(device)
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    model.eval()
    decoder.eval()

    with torch.no_grad():

        z = model(
            data.x,
            data.edge_index
        )

        positive_scores = torch.sigmoid(
            decoder(
                z,
                positive_index
            )
        )

        negative_scores = torch.sigmoid(
            decoder(
                z,
                negative_index
            )
        )

        positive_scores_reversed = torch.sigmoid(
            decoder(
                z,
                positive_reversed
            )
        )

        negative_scores_reversed = torch.sigmoid(
            decoder(
                z,
                negative_reversed
            )
        )

    # --------------------------------------------------------
    # NumPy
    # --------------------------------------------------------

    positive_scores = (
        positive_scores
        .cpu()
        .numpy()
    )

    negative_scores = (
        negative_scores
        .cpu()
        .numpy()
    )

    positive_scores_reversed = (
        positive_scores_reversed
        .cpu()
        .numpy()
    )

    negative_scores_reversed = (
        negative_scores_reversed
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Original metrics
    # --------------------------------------------------------

    labels = (
        [1] * len(positive_scores)
        +
        [0] * len(negative_scores)
    )

    scores = (
        list(positive_scores)
        +
        list(negative_scores)
    )

    auc = roc_auc_score(
        labels,
        scores
    )

    ap = average_precision_score(
        labels,
        scores
    )

    # --------------------------------------------------------
    # Reversed metrics
    # --------------------------------------------------------

    reversed_scores = (
        list(positive_scores_reversed)
        +
        list(negative_scores_reversed)
    )

    reversed_auc = roc_auc_score(
        labels,
        reversed_scores
    )

    reversed_ap = average_precision_score(
        labels,
        reversed_scores
    )

    # --------------------------------------------------------
    # Score differences
    # --------------------------------------------------------

    positive_difference = abs(
        positive_scores
        -
        positive_scores_reversed
    )

    negative_difference = abs(
        negative_scores
        -
        negative_scores_reversed
    )

    max_difference = max(
        positive_difference.max(),
        negative_difference.max()
    )

    mean_difference = (
        positive_difference.mean()
        +
        negative_difference.mean()
    ) / 2

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("ORIENTATION SENSITIVITY")
    print("=" * 60)

    print()
    print("Original:")
    print(
        f"AUC: {auc:.6f}"
    )
    print(
        f"AP:  {ap:.6f}"
    )

    print()
    print("Reversed:")
    print(
        f"AUC: {reversed_auc:.6f}"
    )
    print(
        f"AP:  {reversed_ap:.6f}"
    )

    print()
    print("Differences:")

    print(
        f"AUC difference: "
        f"{abs(auc - reversed_auc):.10f}"
    )

    print(
        f"AP difference:  "
        f"{abs(ap - reversed_ap):.10f}"
    )

    print(
        f"Mean score difference: "
        f"{mean_difference:.10f}"
    )

    print(
        f"Max score difference:  "
        f"{max_difference:.10f}"
    )

    if max_difference < 1e-6:

        print()
        print(
            "✓ Il decoder è simmetrico rispetto "
            "all'orientamento dell'arco."
        )

    else:

        print()
        print(
            "⚠ Il decoder presenta sensibilità "
            "all'orientamento dell'arco."
        )

    return {
        "auc": auc,
        "ap": ap,
        "reversed_auc": reversed_auc,
        "reversed_ap": reversed_ap,
        "auc_difference": abs(
            auc - reversed_auc
        ),
        "ap_difference": abs(
            ap - reversed_ap
        ),
        "mean_score_difference": mean_difference,
        "max_score_difference": max_difference
    }