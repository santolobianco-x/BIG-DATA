"""
MODULO DI CONFIGURAZIONE DEL PROGETTO
-> contiene tutti i percorsi dei file, iperparametri e configurazioni varie
"""
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent #percorso assoluto della cartella

DATASET_PATH = PROJECT_ROOT / "data" / "ca-AstroPh.txt.gz"
DATASET_NAME = DATASET_PATH.name.split(".")[0]
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

GNN_SEEDS = (1, 2) #Seed multipli per valutare la stabilità della GNN.
#con più seed si può calcolare media e deviazione standard


# ============================================================
# DATA SPLIT
# ============================================================

TEST_RATIO = 0.10 #10% degli archi va nel test set
VAL_RATIO = 0.10 #10% degli archi va nel validation set

SUPERVISION_RATIO = 0.10 #10% utilizzato per calcolare la loss nel decoder


# ============================================================
# GNN
# ============================================================

HIDDEN_CHANNELS = 64 # dimensione layer nascosti in GraphSAGE
OUT_CHANNELS = 32 # dimensione dell'embedding dei nodi
DECODER_HIDDEN_DIM = 64 # dimensione del layer nascosto nel decoder MLP

LEARNING_RATE = 0.001
EPOCHS = 500

DEVICE = "cpu" 


# ============================================================
# HITS@K
# ============================================================

NUM_QUERY_POSITIVES = 1000 # Quante query positive valutare
NEGATIVES_PER_QUERY = 999 # Quanti falsi archi associare a ogni arco vero

HITS_K_VALUES = (10, 50, 100) # quante volte il vero link è in Top-10


# ============================================================
# FEATURE ABLATION
# ============================================================
"""
L'ablazione permette di capire quanto ogni pezzo del sistema contribuisca
alle perfomance finali.
Si analizzano 4 casi:
- full --> modello completo(tutte le informazioni disponibili)
- degree --> si toglie il clustering per verificare se le perfomance rimangono uguali
- clustering --> si toglie il degree per verificare se le perfomance vengono alterate o meno
- constant --> si da a tutti i nodi una feature fissa e si forza ad utilizzare solo la struttura degli archi"""
ABLATION_EPOCHS = 400
ABLATION_SEED = 1
ABLATION_LEARNING_RATE = 0.01

