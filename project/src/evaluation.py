import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def _to_list(values):
    """
    Converte tensor, NumPy array o lista in una lista Python.
    """

    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()

    if isinstance(values, np.ndarray):
        values = values.tolist()

    return list(values)


def evaluate_scores(pos_scores, neg_scores):
    """
    Calcola ROC-AUC e Average Precision.

    Parameters
    ----------
    pos_scores : iterable
        Score assegnati agli archi positivi.

    neg_scores : iterable
        Score assegnati agli archi negativi.

    Returns
    -------
    dict
        Dizionario contenente AUC e AP.
    """

    pos_scores = _to_list(pos_scores)
    neg_scores = _to_list(neg_scores)

    y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
    y_scores = pos_scores + neg_scores

    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    return {
        "auc": auc,
        "ap": ap,
    }


def hits_at_k(query_positive_scores, query_negative_scores, k):
    """
    Calcola Hits@K in modalità query-based.

    Ogni query contiene:
        - un arco positivo
        - uno o più archi negativi candidati

    Il vero arco è considerato un hit se il suo rank
    nella classifica dei candidati è <= K.

    Parameters
    ----------
    query_positive_scores : iterable
        Score del vero arco per ogni query.

        Esempio:
            [0.91, 0.73, 0.42, ...]

    query_negative_scores : iterable of iterables
        Score degli archi negativi per ogni query.

        Esempio:
            [
                [0.81, 0.32, 0.15, ...],
                [0.91, 0.62, 0.51, ...],
                [0.30, 0.21, 0.18, ...],
                ...
            ]

    k : int
        Numero massimo di posizioni considerate.

    Returns
    -------
    float
        Percentuale di query per cui il positivo
        compare entro le prime K posizioni.
    """

    if k <= 0:
        raise ValueError("k deve essere maggiore di 0.")

    query_positive_scores = _to_list(query_positive_scores)

    if len(query_positive_scores) != len(query_negative_scores):
        raise ValueError(
            "Il numero di positive scores deve coincidere "
            "con il numero di query negative."
        )

    if len(query_positive_scores) == 0:
        return 0.0

    hits = 0

    for positive_score, negative_scores in zip(
        query_positive_scores,
        query_negative_scores
    ):

        negative_scores = _to_list(negative_scores)

        # Il rank del positivo è:
        #
        # 1 + numero di negativi con score maggiore
        #
        rank = 1 + sum(
            score > positive_score
            for score in negative_scores
        )

        if rank <= k:
            hits += 1

    return hits / len(query_positive_scores)