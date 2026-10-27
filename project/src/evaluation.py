"""
MODULO CONTENENTE LE METRICHE DI VALUTAZIONE MATEMATICHE
--> Calcola le metriche globali di classificazione binaria e la metrica di ranking
simulando un sistema di raccomandazione.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def to_list(values):
    """
    Standardizza i formati dei dati.
    Nel progetto si hanno array generati da scikit-learn e tensori generati dalla
    rete neurale.
    Questa funzione fa da ponte scaricando i tensori dalla GPU(se presente)
    e convertendo tutto in un formato standard Python.
    """

    # Se è un tensore Pytorch
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()

    # Se è un array numpy
    if isinstance(values, np.ndarray):
        return values.tolist()

    # Se è già un iterabile Python
    return list(values)


def evaluate_scores(pos_scores, neg_scores):
    """
    Calcola le metriche di valutazione globali: ROC-AUC e Average Precision(AP).
    - AUC(Area Under the ROC Curve): è la probabilita che il modello assegni uno 
    score più alto ad un arco vero casuale rispetto ad uno falso.
    - AP (Average Precision): Riassume la curva Precision-Recall. È molto più
    rigorosa e affidabile dell'AUC in scenari altamente sbilanciati
    """

    # Standardizzazione in array NumPy piatti(1D) per scikit-learn
    pos_scores = np.asarray(to_list(pos_scores), dtype=np.float64).reshape(-1)
    neg_scores = np.asarray(to_list(neg_scores), dtype=np.float64).reshape(-1)

    
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        raise ValueError(
            "Sono necessari almeno uno score positivo e uno negativo."
        )

    if not np.all(np.isfinite(pos_scores)) or not np.all(np.isfinite(neg_scores)):
        raise ValueError("Gli score devono essere tutti finiti.")


    # Creazione dell'etichette:
    # * 1 a tutti i pos_scores
    # * 0 a tutti i neg_scores
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
    } # Dizionario contenente i valori float di AUC e AP


def hits_at_k(query_positive_scores, query_negative_scores, k, tie="pessimistic"):
    """
    Calcola Hits@K in modalità query-based.
    La metrica Hits@K prevede la ricerca di un link in mezzo a centinaia di link falsi,
    Il modellol è in grado di mettere il link vero tra le prime k posizioni della classifica?
    """

    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k deve essere un intero maggiore di 0.")

    if tie not in {"pessimistic", "average", "optimistic"}:
        raise ValueError(
            "tie deve essere 'pessimistic', 'average' oppure 'optimistic'."
        )
    # tie serve a decidere se il modello deve o non deve penalizzare l'arco vero

    # Formattazione array
    positive_scores = np.asarray(
        to_list(query_positive_scores),
        dtype=np.float64,
    ).reshape(-1)

    negative_rows = [
        np.asarray(to_list(scores), dtype=np.float64).reshape(-1)
        for scores in query_negative_scores
    ]

    # Controlli di integrità dimensionale
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

    
    candidate_counts = {len(row) for row in negative_rows}

    if len(candidate_counts) != 1:
        raise ValueError(
            "Tutte le query devono avere lo stesso numero "
            "di candidati negativi."
        )

    negative_scores = np.stack(negative_rows, axis=0)

    # VETTORIZZAZIONE: si contano quanti negativi hanno lo score strettamente maggiore
    # del positivo corrispondente. Si allinea l'array alla matrice 2d
    greater = np.sum(
        negative_scores > positive_scores[:, None],
        axis=1,
    )

    # Si trovano i negativi che sono in pareggio con il positivo
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

    # Se il rank è minore o uguale a K è un'hit, e si calcola la media gbloale.
    return float(np.mean(rank <= k)) # percentuale di query in cui il modello ha fatto centro
