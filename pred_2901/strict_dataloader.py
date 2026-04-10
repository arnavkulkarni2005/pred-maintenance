import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import RobustScaler
import warnings
import config

warnings.filterwarnings("ignore")

def get_file_paths():
    """Returns a list of Normal files and a dictionary of Anomaly files."""
    normal_dir = config.DATASET_PATH / '0'
    normal_files = list(normal_dir.rglob("*.parquet"))
    
    anomaly_files = {}
    for cls in range(1, 9):
        cls_dir = config.DATASET_PATH / str(cls)
        if cls_dir.exists():
            anomaly_files[cls] = list(cls_dir.rglob("*.parquet"))
            
    return normal_files, anomaly_files

def process_and_fit_preprocessors(train_files):
    """Fits Imputer and Scaler strictly on the provided training files."""
    print("Fitting Imputer and Scaler on Train Files only...")
    all_train_data = []
    
    for f in train_files:
        try:
            df = pd.read_parquet(f)
            for col in config.TARGET_SENSORS:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[config.TARGET_SENSORS]
            all_train_data.append(df.values)
        except Exception:
            continue
            
    # Concatenate to fit global fold transforms
    concatenated = np.vstack(all_train_data)
    
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
    scaler = RobustScaler()
    
    print("Running Bayesian Imputation fit...")
    imputed = imputer.fit_transform(concatenated)
    print("Running RobustScaler fit...")
    
    # THE CRITICAL FIX: The scaler learns 'Normal' from the train files
    scaler.fit(imputed) 
    
    return imputer, scaler

def process_files_to_windows(file_paths, imputer, scaler, window_size, stride):
    """Applies the FOLD'S GLOBAL transformations to the files."""
    windows = []
    
    for f in file_paths:
        try:
            df = pd.read_parquet(f)
            for col in config.TARGET_SENSORS:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[config.TARGET_SENSORS]
            
            # Impute using the fold's global rules
            imputed = imputer.transform(df.values)
            
            # THE CRITICAL FIX: Scale using the fold's Normal ruler. 
            # This allows anomalies to actually look huge to the model.
            scaled = scaler.transform(imputed)
            
            # Clip hardware extremes
            clean_data = np.clip(scaled, -15.0, 15.0)
            
            if len(clean_data) < window_size:
                continue
                
            for i in range(0, len(clean_data) - window_size + 1, stride):
                windows.append(clean_data[i:i+window_size])
        except Exception:
            continue
            
    if not windows:
        return torch.empty(0)
        
    return torch.tensor(np.array(windows, dtype=np.float32))