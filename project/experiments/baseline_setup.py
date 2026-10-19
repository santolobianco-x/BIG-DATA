
"""Configurazione dei metodi baseline per la link prediction."""

from src.baseline import (
    common_neighbors_scores,
    jaccard_scores,
    adamic_adar_scores,
    preferential_attachment_scores,
)


def get_baseline_methods():
    """Restituisce i metodi baseline disponibili."""
    return {
        "Common Neighbors": common_neighbors_scores,
        "Jaccard": jaccard_scores,
        "Adamic-Adar": adamic_adar_scores,
        "Preferential Attachment": preferential_attachment_scores,
    }

