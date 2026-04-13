"""
strict_dataloader.py

Provides file-level data loading with no leakage between folds.
Key design: imputer and scaler are ALWAYS fitted only on the training
files of a given fold and never see validation or anomaly data before
transform time.

Public API
----------
get_file_paths()
    -> (normal_files, anomaly_files_dict)

process_and_fit_preprocessors(train_files)
    -> (imputer, scaler)

process_files_to_windows(file_paths, imputer, scaler, window_size, stride)
    -> torch.Tensor of shape [N, W, F]

save_preprocessors(imputer, scaler, path)
load_preprocessors(path)
    -> (imputer, scaler)
"""

import warnings
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import RobustScaler

import config

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ #
#  File discovery                                                      #
# ------------------------------------------------------------------ #

def get_file_paths():
    """
    Scans the dataset directory and returns:
      - normal_files : list of Paths for class 0 (normal)
      - anomaly_files_dict : {class_int: [Path, ...], ...} for classes 1-8

    Respects config.MAX_FILES_PER_CLASS when set.
    """
    normal_dir = config.DATASET_PATH / '0'
    if not normal_dir.exists():
        raise FileNotFoundError(f"Normal data directory not found: {normal_dir}")

    normal_files = list(normal_dir.rglob("*.parquet"))
    if config.MAX_FILES_PER_CLASS:
        normal_files = normal_files[:config.MAX_FILES_PER_CLASS]

    anomaly_files = {}
    for cls in config.ANOMALY_CLASSES:
        cls_dir = config.DATASET_PATH / str(cls)
        if cls_dir.exists():
            files = list(cls_dir.rglob("*.parquet"))
            if config.MAX_FILES_PER_CLASS:
                files = files[:config.MAX_FILES_PER_CLASS]
            if files:
                anomaly_files[cls] = files

    print(
        f"Files found — normal: {len(normal_files)}, "
        f"anomaly: {sum(len(v) for v in anomaly_files.values())} "
        f"across {len(anomaly_files)} classes."
    )
    return normal_files, anomaly_files


# ------------------------------------------------------------------ #
#  Preprocessing                                                       #
# ------------------------------------------------------------------ #

def _load_raw_values(file_paths, max_files=None):
    """
    Loads and stacks raw sensor values from a list of parquet files.
    Missing sensor columns are filled with NaN.
    Returns a float32 numpy array of shape [total_rows, n_sensors].
    """
    files = file_paths[:max_files] if max_files else file_paths
    chunks = []

    for f in files:
        try:
            df = pd.read_parquet(f)
            for col in config.TARGET_SENSORS:
                if col not in df.columns:
                    df[col] = np.nan
            arr = df[config.TARGET_SENSORS].values.astype(np.float32)
            # Replace any non-finite values with NaN for the imputer
            arr = np.where(np.isfinite(arr), arr, np.nan)
            chunks.append(arr)
        except Exception:
            continue

    if not chunks:
        raise ValueError("No valid files could be loaded from the provided list.")

    return np.vstack(chunks)


def process_and_fit_preprocessors(train_files):
    """
    Fits an IterativeImputer (BayesianRidge) and a RobustScaler
    strictly on the provided training files.

    To keep fit time tractable on large datasets, only up to
    config.MAX_FIT_FILES files are used for fitting; the rest of
    the training files are still windowed and used for training.

    Returns
    -------
    imputer : fitted IterativeImputer
    scaler  : fitted RobustScaler
    """
    print(f"  Fitting preprocessors on ≤{config.MAX_FIT_FILES} of "
          f"{len(train_files)} training files...")

    fit_data = _load_raw_values(train_files, max_files=config.MAX_FIT_FILES)

    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=10,
        random_state=42,
        verbose=0,
    )
    scaler = RobustScaler()

    print("    Running BayesianRidge imputation fit...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imputed = imputer.fit_transform(fit_data)

    print("    Running RobustScaler fit...")
    scaler.fit(imputed)

    return imputer, scaler


# ------------------------------------------------------------------ #
#  Windowing                                                           #
# ------------------------------------------------------------------ #

def process_files_to_windows(file_paths, imputer, scaler,
                              window_size=None, stride=None):
    """
    Applies the pre-fitted imputer and scaler to each file, then
    slices the resulting time-series into overlapping windows.

    Parameters
    ----------
    file_paths  : list of Path / str
    imputer     : fitted IterativeImputer
    scaler      : fitted RobustScaler
    window_size : int, defaults to config.WINDOW_SIZE
    stride      : int, defaults to config.STRIDE

    Returns
    -------
    torch.Tensor of shape [N, window_size, n_sensors], dtype float32.
    Returns torch.empty(0) if no valid windows could be created.
    """
    if window_size is None:
        window_size = config.WINDOW_SIZE
    if stride is None:
        stride = config.STRIDE

    windows = []

    for f in file_paths:
        try:
            df = pd.read_parquet(f)
            for col in config.TARGET_SENSORS:
                if col not in df.columns:
                    df[col] = np.nan

            raw = df[config.TARGET_SENSORS].values.astype(np.float32)
            raw = np.where(np.isfinite(raw), raw, np.nan)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imputed = imputer.transform(raw)

            scaled = scaler.transform(imputed)
            clean  = np.clip(scaled, -config.CLIP_VALUE, config.CLIP_VALUE)

            if len(clean) < window_size:
                continue

            for i in range(0, len(clean) - window_size + 1, stride):
                windows.append(clean[i: i + window_size])

        except Exception:
            continue

    if not windows:
        return torch.empty(0)

    return torch.tensor(np.array(windows, dtype=np.float32))


# ------------------------------------------------------------------ #
#  Save / load preprocessors                                           #
# ------------------------------------------------------------------ #

def save_preprocessors(imputer, scaler, path):
    """
    Saves the fitted imputer and scaler to disk using joblib.
    Files: <path>/imputer.joblib and <path>/scaler.joblib
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, path / 'imputer.joblib')
    joblib.dump(scaler,  path / 'scaler.joblib')
    print(f"  Preprocessors saved to {path}")


def load_preprocessors(path):
    """
    Loads a previously saved imputer and scaler from disk.

    Returns
    -------
    (imputer, scaler)
    """
    path = Path(path)
    imputer_path = path / 'imputer.joblib'
    scaler_path  = path / 'scaler.joblib'

    if not imputer_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"Preprocessor files not found in {path}. "
            "Run master_pipeline.py first to generate them."
        )

    imputer = joblib.load(imputer_path)
    scaler  = joblib.load(scaler_path)
    return imputer, scaler