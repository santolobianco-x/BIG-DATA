"""
MODULO DEDICATO AI CONTROLLI DI INTEGRITA' SUI DATI

--> esegue verifiche matematiche rigorose tramite l'uso della teoria degli insieme
per evitare Data Leakage
"""
import torch
import networkx as nx

# ============================================================
# EDGE UTILITIES
# ============================================================

def canonical_edge(u, v):
    return tuple(sorted((u, v))) # restituisce un arco in forma canonica o ordinata


def canonical_edges(edges):
    return {
        canonical_edge(u, v)
        for u, v in edges
    } # converte una lista di archi in un set di archi canonici


# ============================================================
# POSITIVE EDGE SANITY CHECK
# ============================================================

def verify_positive_split(G, train_graph, val_edges, test_edges):
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
        print("Positive split valid")
    else:
        print("Positive split NOT valid")

    return valid


# ============================================================
# NEGATIVE EDGE SANITY CHECK
# ============================================================
def verify_negative_edges(
    G, train_graph, train_negative_edges, val_negative_edges,
    test_negative_edges, val_edges, test_edges
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

    original_edges = canonical_edges(G.edges())
    train_positive = canonical_edges(train_graph.edges())
    val_positive = canonical_edges(val_edges)
    test_positive = canonical_edges(test_edges)

    train_negative = canonical_edges(train_negative_edges)
    val_negative = canonical_edges(val_negative_edges)
    test_negative = canonical_edges(test_negative_edges)

    print(f"Train negatives : {len(train_negative)}")
    print(f"Validation negatives : {len(val_negative)}")
    print(f"Test negatives : {len(test_negative)}")

    # --------------------------------------------------------
    # DISJOINTNESS CHECKS(CONTROLLI DI DISGIUNZIONE)
    # --------------------------------------------------------
    print("\nDisjointness check results:")
    
    checks = {
        "Train negatives ∩ Original == ∅": train_negative.isdisjoint(original_edges),
        "Validation negatives ∩ Original == ∅": val_negative.isdisjoint(original_edges),
        "Test negatives ∩ Original == ∅": test_negative.isdisjoint(original_edges),
        
        "Train ∩ Validation negatives == ∅": train_negative.isdisjoint(val_negative),
        "Train ∩ Test negatives == ∅": train_negative.isdisjoint(test_negative),
        "Validation ∩ Test negatives == ∅": val_negative.isdisjoint(test_negative),

        "Train neg ∩ Train pos == ∅": train_negative.isdisjoint(train_positive),
        "Train neg ∩ Val pos == ∅": train_negative.isdisjoint(val_positive),
        "Train neg ∩ Test pos == ∅": train_negative.isdisjoint(test_positive),
        "Val neg ∩ Val pos == ∅": val_negative.isdisjoint(val_positive),
        "Test neg ∩ Test pos == ∅": test_negative.isdisjoint(test_positive),
    }

    for description, passed in checks.items():
        print(f"{'✓' if passed else '✗'} {description}")

    valid = all(checks.values())

    print()
    if valid:
        print("Negative sampling valid")
    else:
        print("Negative sampling NOT valid")

    return valid

# ============================================================
# ISOLATED NODE ANALYSIS
# ============================================================

def analyze_isolated_nodes(train_graph, val_edges, test_edges, graph_name="train"):
    """
    Analizza quanti nodi diventano isolati nel training graph a causa 
    dell'edge split e quanti archi di validation/test coinvolgono tali nodi.
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
        f"Isolated nodes in {graph_name} : "
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
    - grado normalizzato;
    - clustering coefficient.
    """

    nodes = list(G.nodes())

    degrees = [G.degree(node) for node in nodes]

    max_degree = max(degrees)

    clustering = { node: value for node, value in nx.clustering(G).items()}

    features = torch.tensor(
        [[degree / max_degree, clustering[node]]
            for node, degree
            in zip(nodes, degrees)
        ],
        dtype=torch.float
    )

    return features #contiene sia il grado normalizzato e sia il clustering


# ============================================================
# FEATURE SOURCE CHECK
# ============================================================

def verify_feature_source(G, train_graph, data):
    """
    Verifica che le feature utilizzate dalla GNN siano calcolate dal training 
    graph e non dal grafo completo.

    Se le feature fossero calcolate sul grafo originale, passerebbero informazioni
    sui test edges (i quali esistono nell'originale) falsando le predizioni.
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

    actual_features = data.x.cpu() # prende i tensori delle feature reali che la pipeline ha appena costruito

    max_difference_original = torch.max(torch.abs( actual_features - original_features)).item()
    #se è uguale l'errore massimo tra i valori allora l'errore è 0

    max_difference_train = torch.max(torch.abs( actual_features - train_features)).item()

    matches_original = torch.allclose(
        actual_features,
        original_features
    ) # verifica se sono vicini, ignorando decimali piccolissimi

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
            "Features match the training graph."
        )
    else:
        print(
            "Features do NOT match the training graph."
        )

    return matches_train


def verify_gnn_split(
    train_graph,
    message_passing_graph,
    message_passing_edges,
    supervised_train_edges,
    supervised_train_negative_edges,
    val_negative_edges,
    test_negative_edges,
    val_edges,
    test_edges,
):
    """
    Verifica la correttezza dello split utilizzato dalla GNN.

    Controlla che:
    - message passing e supervisione siano disgiunti;
    - la loro unione ricostruisca il training graph;
    - validation e test non siano nel grafo di message passing;
    - i negativi di supervisione non appartengano al grafo originale
    del training e siano disgiunti dai negativi di validation e test.
    """
    print()
    print("=" * 60)
    print("GNN SPLIT SANITY CHECK")
    print("=" * 60)

    train_set = canonical_edges(train_graph.edges())
    mp_set = canonical_edges(message_passing_edges)
    supervised_set = canonical_edges(supervised_train_edges)
    graph_mp_set = canonical_edges(message_passing_graph.edges())
    val_set = canonical_edges(val_edges)
    test_set = canonical_edges(test_edges)
    
    supervised_neg_set = canonical_edges(supervised_train_negative_edges)
    val_neg_set = canonical_edges(val_negative_edges)
    test_neg_set = canonical_edges(test_negative_edges)

    # Eseguiamo i controlli in modo ottimizzato senza allocare nuovi set
    checks = {
        "Message-passing graph consistent": graph_mp_set == mp_set,
        "Message passing and supervision edges disjoint": mp_set.isdisjoint(supervised_set),
        "Union equals training graph": (mp_set | supervised_set) == train_set,
        "No validation edges in message passing": mp_set.isdisjoint(val_set),
        "No test edges in message passing": mp_set.isdisjoint(test_set),
        "Supervised negatives absent from train positives": supervised_neg_set.isdisjoint(train_set),
        "Supervised negatives disjoint from validation negatives": supervised_neg_set.isdisjoint(val_neg_set),
        "Supervised negatives disjoint from test negatives": supervised_neg_set.isdisjoint(test_neg_set),
    }

    for description, passed in checks.items():
        print(f"{'✓' if passed else '✗'} {description}")

    valid = all(checks.values())

    print()
    print(
        "GNN split valid"
        if valid
        else "GNN split NOT valid"
    )

    return valid