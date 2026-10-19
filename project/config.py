
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_ROOT / "data" / "ca-AstroPh.txt.gz"
DATASET_NAME = DATASET_PATH.name.split(".")[0]
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

GNN_SEEDS = (1, 2)


# ============================================================
# DATA SPLIT
# ============================================================

TEST_RATIO = 0.10
VAL_RATIO = 0.10

SUPERVISION_RATIO = 0.10


# ============================================================
# GNN
# ============================================================

HIDDEN_CHANNELS = 64
OUT_CHANNELS = 32
DECODER_HIDDEN_DIM = 64

LEARNING_RATE = 0.001
EPOCHS = 500

DEVICE = "cpu"


# ============================================================
# HITS@K
# ============================================================

NUM_QUERY_POSITIVES = 1000
NEGATIVES_PER_QUERY = 999

HITS_K_VALUES = (10, 50, 100)


# ============================================================
# FEATURE ABLATION
# ============================================================

ABLATION_EPOCHS = 400
ABLATION_SEED = 1
ABLATION_LEARNING_RATE = 0.01

