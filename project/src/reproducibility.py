"""
MODULO PER GARANTIRE LA RIPRODUCIBILITA' DEGLI ESPERIMENTI
--> vengono fissati i seed dei generatori di numeri pseudo-casuali
per ottenere risultati deterministici.
"""

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # inizializzazione dei pesi della GNN e del Decoder



    """Se la macchina su cui è eseguito il codice è dotata di scheda video dedicata
    fissa il seed anche per i tensori allocati sulla memoria della GPU"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

