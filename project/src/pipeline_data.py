"""
MODULO PER L'ORCHESTRAZIONE E LA PREPARAZIONE DEI DATI SPERIMENTALI

--> coordina la suddivisione del grafo in set di addestramento, validazione e test.
Gestisce il campionamento degli archi negativi per evitare qualsiasi forma di 
data leakage(sovrapposzione di dati tra i set).
"""

from config import (
    SEED,
    TEST_RATIO,
    VAL_RATIO,
    SUPERVISION_RATIO,
    NUM_QUERY_POSITIVES,
    NEGATIVES_PER_QUERY,
)

from src.split import split_edges, split_train_edges_for_gnn
from src.negative_sampling import (
    generate_negative_edges,
    generate_query_negative_edges,
)


def prepare_graph_splits(graph):
    """
    Coordina la suddivisione del grafo a due livelli:
    - train, validation e test set
    - message passing(struttura) e supervision(etichette)
    """

    
    train_graph, val_edges, test_edges = split_edges(
        graph,
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    train_edges = list(train_graph.edges())


    """
    Il Message Passing è l'algoritmo dove:
        - ogno nodo guarda i suo vicini diretti nel grafo e raccoglie le informazioni
        - il nodo prende i messaggi ricevuti dai vicini e ne fa una media
        - unisce le informazioni ricevute alle informazioni di partenza
    """
    (message_passing_graph,message_passing_edges,supervised_train_edges,) = split_train_edges_for_gnn(
        train_graph,
        supervision_ratio=SUPERVISION_RATIO,
        seed=SEED,
    )
    """
    Per evitare che la GNN bari, si nasconde una percentuale degli archi di training.
    La GNN userà il message_passing_graph per calcolare gli embedding e cercherà di indovinare
    i supervised_train_edges.
    """
    

    print("\n" + "=" * 70)
    print("EDGE SPLIT")
    print("=" * 70)
    print(f"Original edges:            {graph.number_of_edges()}")
    print(f"Train edges:               {len(train_edges)}")
    print(f"Validation edges:          {len(val_edges)}")
    print(f"Test edges:                {len(test_edges)}")
    print(f"Message-passing edges:     {len(message_passing_edges)}")
    print(f"Supervised positive edges: {len(supervised_train_edges)}")

    return {
        "train_graph": train_graph,
        "train_edges": train_edges,
        "val_edges": val_edges,
        "test_edges": test_edges,
        "message_passing_graph": message_passing_graph,
        "message_passing_edges": message_passing_edges,
        "supervised_train_edges": supervised_train_edges,
    } # dizionario completo contenente tutti i sottografi e le liste di archi divisi


def prepare_negative_edges(graph, splits):
    """
    Genera gli archi negativi.
    Ogni volta che viene generato un set di negativi, questo viene aggiunto all'insieme delle coppie
    proibite. 
    Questo garantisce che non ci sia mai un'intersezione tra i negativi e i positivi.
    """
    print("\n" + "=" * 70)
    print("NEGATIVE SAMPLING")
    print("=" * 70)

    train_edges = splits["train_edges"]
    val_edges = splits["val_edges"]
    test_edges = splits["test_edges"]
    supervised_train_edges = splits["supervised_train_edges"]

    original_edges = {
        tuple(sorted(edge))
        for edge in graph.edges()
    }

    train_negative_edges = generate_negative_edges(
        graph,
        num_edges=len(train_edges),
        seed=SEED,
        forbidden_edges=original_edges,
    )


    # insieme degli archi vietati: archi positivi e negativi di train
    train_forbidden = original_edges | {
        tuple(sorted(edge))
        for edge in train_negative_edges
    }


    val_negative_edges = generate_negative_edges(
        graph,
        num_edges=len(val_edges),
        seed=SEED + 1,
        forbidden_edges=train_forbidden,
    )

    # insieme degli archi vietati: archi positivi, negativi di train e validation
    val_forbidden = train_forbidden | {
        tuple(sorted(edge))
        for edge in val_negative_edges
    }

    test_negative_edges = generate_negative_edges(
        graph,
        num_edges=len(test_edges),
        seed=SEED + 2,
        forbidden_edges=val_forbidden,
    )

    # insieme degli archi vietati: archi positivi, negativi di train, validation e test
    test_forbidden = val_forbidden | {
        tuple(sorted(edge))
        for edge in test_negative_edges
    }

    supervised_train_negative_edges = generate_negative_edges(
        graph,
        num_edges=len(supervised_train_edges),
        seed=SEED + 100,
        forbidden_edges=test_forbidden,
    )

    print(f"Train negatives:           {len(train_negative_edges)}")
    print(f"Validation negatives:      {len(val_negative_edges)}")
    print(f"Test negatives:            {len(test_negative_edges)}")
    print(f"GNN supervised negatives:  {len(supervised_train_negative_edges)}")

    return {
        "train_negative_edges": train_negative_edges,
        "val_negative_edges": val_negative_edges,
        "test_negative_edges": test_negative_edges,
        "supervised_train_negative_edges": supervised_train_negative_edges,
    } # dizionario contenente le liste di archi negativi per ogni fase


def prepare_hits_queries(graph, test_edges):
    """
    Prepara le query per la metrica di ranking Hits@K.

    Hits@K simula un sistema di raccomandazione dove prende 1 arco positivo
    e lo nasconde in mezzo a N archi distrattori. Se il modello assegna all'arco vero
    uno score più alto rispetto ai distrattori, il modello funziona.
    """
    print("\n" + "=" * 70)
    print("HITS@K QUERY GENERATION")
    print("=" * 70)

    if len(test_edges) < NUM_QUERY_POSITIVES:
        raise ValueError(
            "Il test set contiene meno archi delle query richieste."
        )

    query_positive_edges = list(test_edges[:NUM_QUERY_POSITIVES])

    # Pe ognuno dei bersagli si generano centinaia di distrattori
    query_negative_edges = generate_query_negative_edges(
        graph,
        query_positive_edges,
        num_negatives_per_query=NEGATIVES_PER_QUERY,
        seed=SEED,
    )

    print(f"Queries: {len(query_positive_edges)}")
    print(f"Negative candidates/query: {NEGATIVES_PER_QUERY}")
    print(f"Candidates/query: {1 + NEGATIVES_PER_QUERY}")
    print(
        "Total negative candidates: "
        f"{sum(len(edges) for edges in query_negative_edges)}"
    )

    return query_positive_edges, query_negative_edges #tuple contenenti query positive e negative

