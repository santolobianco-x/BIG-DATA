
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def to_list(values):
    """
    Converte un tensor PyTorch, un array NumPy o un iterabile
    in una lista Python.
    """

    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()

    if isinstance(values, np.ndarray):
        return values.tolist()

    return list(values)


def evaluate_scores(pos_scores, neg_scores):
    """
    Calcola ROC-AUC e Average Precision.

    Parameters
    ----------
    pos_scores : iterable
        Score degli archi positivi.

    neg_scores : iterable
        Score degli archi negativi.

    Returns
    -------
    dict
        Dizionario contenente AUC e AP.
    """

    pos_scores = np.asarray(to_list(pos_scores), dtype=np.float64).reshape(-1)
    neg_scores = np.asarray(to_list(neg_scores), dtype=np.float64).reshape(-1)

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        raise ValueError(
            "Sono necessari almeno uno score positivo e uno negativo."
        )

    if not np.all(np.isfinite(pos_scores)) or not np.all(np.isfinite(neg_scores)):
        raise ValueError("Gli score devono essere tutti finiti.")

    y_true = np.concatenate([
        np.ones(len(pos_scores), dtype=np.int8),
        np.zeros(len(neg_scores), dtype=np.int8),
    ])

    y_scores = np.concatenate([pos_scores, neg_scores])

    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    return {
        "auc": float(auc),
        "ap": float(ap),
    }


def hits_at_k(query_positive_scores, query_negative_scores, k, tie="pessimistic"):
    """
    Calcola Hits@K in modalità query-based.

    Ogni query contiene un positivo e uno o più candidati negativi.

    Parametri
    ---------
    query_positive_scores : iterable
        Score del vero arco per ogni query.

    query_negative_scores : iterable of iterables
        Score dei candidati negativi per ogni query.

    k : int
        Numero massimo di posizioni considerate.

    tie : str
        Politica per gli score a pari merito:
        - "pessimistic": tutti i negativi a pari score precedono il positivo;
        - "average": metà dei negativi a pari score precede il positivo;
        - "optimistic": nessun negativo a pari score precede il positivo.

    Returns
    -------
    float
        Frazione di query per cui il positivo raggiunge una posizione <= k.
    """

    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k deve essere un intero maggiore di 0.")

    if tie not in {"pessimistic", "average", "optimistic"}:
        raise ValueError(
            "tie deve essere 'pessimistic', 'average' oppure 'optimistic'."
        )

    positive_scores = np.asarray(
        to_list(query_positive_scores),
        dtype=np.float64,
    ).reshape(-1)

    negative_rows = [
        np.asarray(to_list(scores), dtype=np.float64).reshape(-1)
        for scores in query_negative_scores
    ]

    if len(positive_scores) != len(negative_rows):
        raise ValueError(
            "Il numero di score positivi deve coincidere "
            "con il numero di query negative."
        )

    if len(positive_scores) == 0:
        return 0.0

    if any(len(row) == 0 for row in negative_rows):
        raise ValueError("Ogni query deve avere almeno un candidato negativo.")

    if not np.all(np.isfinite(positive_scores)):
        raise ValueError("Gli score positivi devono essere tutti finiti.")

    if not all(np.all(np.isfinite(row)) for row in negative_rows):
        raise ValueError("Gli score negativi devono essere tutti finiti.")

    # Tutte le query devono avere lo stesso numero di candidati.
    candidate_counts = {len(row) for row in negative_rows}

    if len(candidate_counts) != 1:
        raise ValueError(
            "Tutte le query devono avere lo stesso numero "
            "di candidati negativi."
        )

    negative_scores = np.stack(negative_rows, axis=0)

    # Conteggio vettorizzato dei negativi che precedono il positivo.
    greater = np.sum(
        negative_scores > positive_scores[:, None],
        axis=1,
    )

    ties = np.sum(
        negative_scores == positive_scores[:, None],
        axis=1,
    )

    if tie == "pessimistic":
        rank = 1 + greater + ties
    elif tie == "average":
        rank = 1 + greater + 0.5 * ties
    else:
        rank = 1 + greater

    return float(np.mean(rank <= k))
