
"""Utilities per la riproducibilità degli esperimenti."""

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Imposta i seed dei generatori casuali utilizzati dal progetto."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

