"""
MODULO PER LA CONFIGURAZIONE DEI METODI BASELINE PER LA LINK PREDICTION
"""
from src.baseline import (
    common_neighbors_scores,
    jaccard_scores,
    adamic_adar_scores,
    preferential_attachment_scores,
)


def get_baseline_methods():
    """
    Il dizionario restituito funge da interfaccia unificata, dove:
    - la chiave è il nome che comparirà nei log e nei grafici
    - il valore è la funzione matematica effettiva
    """
    return {
        "Common Neighbors": common_neighbors_scores,
        "Jaccard": jaccard_scores,
        "Adamic-Adar": adamic_adar_scores,
        "Preferential Attachment": preferential_attachment_scores,
    }

