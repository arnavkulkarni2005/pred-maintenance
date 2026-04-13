import torch
from pathlib import Path

# ============================================================
# EXPERIMENT TOGGLE
# Set DEBUG_MODE = True for fast local testing.
# Set DEBUG_MODE = False for full overnight production runs.
# ============================================================
DEBUG_MODE = False

# ============================================================
# PATHS — update these to match your environment
# ============================================================
DATASET_PATH = Path('/home/kulkarni/projects/pred_maintenance/dataset')
OUTPUT_DIR   = Path('/home/kulkarni/projects/pred_maintenance/meeting_prep')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SENSOR CONFIGURATION
# Add or remove sensors here. All other code adapts automatically.
# ============================================================
TARGET_SENSORS = [
    'P-PDG',      # Pressure (Bottom Hole)
    'P-TPT',      # Pressure (Temperature Transducer)
    'T-PDG',      # Temperature (Bottom Hole)
    'T-TPT',      # Temperature (Temperature Transducer)
    'P-MON-CKP',  # Pressure (Upstream Choke)
    'T-JUS-CKP',  # Temperature (Downstream Choke)
    'P-JUS-CKGL', # Pressure (Gas Lift)
    'QGL',        # Flow Rate (Gas Lift)
]

# Anomaly class labels in the dataset (subdirectory names)
ANOMALY_CLASSES = list(range(1, 9))  # [1, 2, 3, 4, 5, 6, 7, 8]

# ============================================================
# DATA PREPROCESSING
# ============================================================
WINDOW_SIZE = 240   # timesteps per window (4 min at 1 Hz)
STRIDE      = 30    # stride between consecutive windows

# Caps files loaded per class to save RAM during debug testing.
# Set to None to use all files.
MAX_FILES_PER_CLASS = 10 if DEBUG_MODE else None

# Maximum files used when fitting the imputer/scaler.
# Fitting BayesianRidge on millions of rows is very slow;
# a representative sample of 50 files is sufficient.
MAX_FIT_FILES = 50

# Post-scaling clip range. Values beyond ±CLIP_VALUE IQRs from
# the median are treated as hardware artefacts and capped.
CLIP_VALUE = 15.0

# ============================================================
# HARDWARE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# TRAINING
# ============================================================
BATCH_SIZE   = 512
FINAL_EPOCHS = 3 if DEBUG_MODE else 25

# ============================================================
# OPTUNA HYPERPARAMETER TUNING
# ============================================================
N_TRIALS      = 5  if DEBUG_MODE else 50
TRIAL_EPOCHS  = 3  if DEBUG_MODE else 10

# MedianPruner settings: start pruning after PRUNER_STARTUP_TRIALS
# trials have finished at least PRUNER_WARMUP_STEPS epochs.
PRUNER_STARTUP_TRIALS = 3
PRUNER_WARMUP_STEPS   = 2