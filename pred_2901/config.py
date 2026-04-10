import torch
from pathlib import Path

# ==========================================
# EXPERIMENT TOGGLE
# ==========================================
# Set to True for quick testing (few files, few trials).
# Set to False for the full overnight production run.
DEBUG_MODE = False 

# ==========================================
# PATHS
# ==========================================
DATASET_PATH = Path('/home/kulkarni/projects/pred_maintenance/dataset')
OUTPUT_DIR = Path('/home/kulkarni/projects/pred_maintenance/meeting_prep')

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# SENSOR CONFIGURATION (The "Gatekeeper")
# ==========================================
TARGET_SENSORS = [
    'P-PDG',        # Pressure (Bottom Hole)
    'P-TPT',        # Pressure (Temperature Transducer)
    'T-PDG',        # Temperature (Bottom Hole)
    'T-TPT',        # Temperature (Temperature Transducer)
    'P-MON-CKP',    # Pressure (Upstream Choke)
    'T-JUS-CKP',    # Temperature (Downstream Choke)
    'P-JUS-CKGL',   # Pressure (Gas Lift)
    'QGL'           # Flow Rate (Gas Lift)
]

# ==========================================
# DATA PREPROCESSING PARAMETERS
# ==========================================
WINDOW_SIZE = 240
STRIDE = 30
# Limits files loaded per class to save RAM during testing
MAX_FILES_PER_CLASS = 50 if DEBUG_MODE else None 

# ==========================================
# HARDWARE
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# OPTUNA HYPERPARAMETER TUNING
# ==========================================
N_TRIALS = 3 if DEBUG_MODE else 100
TRIAL_EPOCHS = 5
PRUNER_STARTUP_TRIALS = 5
PRUNER_WARMUP_STEPS = 1

# ==========================================
# FINAL MODEL TRAINING
# ==========================================
FINAL_EPOCHS = 2 if DEBUG_MODE else 25