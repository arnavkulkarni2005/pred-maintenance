import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === CONFIG: FORCE SPECIFIC SENSORS ===
# These are the 8 common sensors found in your inspection (0 missing values)
# We ignore rare sensors to ensure every file produces shape [120, 8]
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

class WindowedDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_data(dataset_path, window_size=120, stride=60):
    print(f"Loading data from: {dataset_path}")
    print(f"Forcing Sensor List ({len(TARGET_SENSORS)}): {TARGET_SENSORS}")
    dataset_path = Path(dataset_path)
    
    # 1. Load Normal Data (Class 0)
    normal_path = dataset_path / '0'
    if not normal_path.exists():
        raise FileNotFoundError(f"Folder not found: {normal_path}")
        
    normal_files = list(normal_path.rglob("*.parquet"))
    print(f"Found {len(normal_files)} Normal files.")
    
    if len(normal_files) == 0:
        raise ValueError("No files found in Class 0!")

    # Split FILES
    train_files, test_files = train_test_split(normal_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42) 
    
    def process_files(file_list, desc):
        windows = []
        skipped = 0
        for f in file_list:
            try:
                df = pd.read_parquet(f)
                
                # --- NEW IMPUTATION LOGIC ---
                # Instead of skipping, we check which sensors are missing and fill them.
                for sensor in TARGET_SENSORS:
                    if sensor not in df.columns:
                        # Create the missing column filled with 0.0
                        df[sensor] = 0.0
                
                # Enforce order: Ensure columns are in the exact order of TARGET_SENSORS
                df_clean = df[TARGET_SENSORS].ffill().bfill()
                # ----------------------------

                # --- SANITIZATION ---
                df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
                df_clean = df_clean.dropna()
                
                vals = df_clean.values
                # Clip to prevent float32 overflow
                vals = np.clip(vals, -1e9, 1e9)
                
                data = vals
                
                if len(data) < window_size:
                    skipped += 1
                    continue

                for i in range(0, len(data) - window_size, stride):
                    windows.append(data[i:i+window_size])
                    
            except Exception:
                skipped += 1
                continue
                
        print(f"[{desc}] Generated {len(windows)} windows. (Skipped {skipped} incompatible files)")
        
        if len(windows) == 0:
            return np.array([])
            
        return np.array(windows, dtype=np.float32)
    # PROCESS
    X_train_raw = process_files(train_files, "Train")
    
    # Check for empty result
    if len(X_train_raw) == 0:
        raise ValueError("CRITICAL: X_train is empty. The files might not contain the required 8 sensors.")

    X_val_raw = process_files(val_files, "Val")
    X_test_normal_raw = process_files(test_files, "Test(Normal)")
    
    # Load Anomalies
    anomaly_windows = []
    print("Processing Anomaly Files...")
    for cls in [2, 3, 4, 5, 6, 7, 8]:
        cls_path = dataset_path / str(cls)
        files = list(cls_path.rglob("*.parquet"))[:50] 
        if files:
            X_anom = process_files(files, f"Class {cls}")
            if len(X_anom) > 0:
                anomaly_windows.append(X_anom)
    
    if anomaly_windows:
        X_test_anom_raw = np.concatenate(anomaly_windows)
    else:
        X_test_anom_raw = np.array([])

    # 2. Normalize
    N, W, F = X_train_raw.shape
    print(f"Training Tensor Shape: {X_train_raw.shape}")
    
    scaler = StandardScaler()
    X_train_flat = X_train_raw.reshape(-1, F)
    scaler.fit(X_train_flat)
    
    def normalize(X):
        if len(X) == 0: return torch.tensor([])
        N, W, F = X.shape
        X_flat = X.reshape(-1, F)
        X_scaled = scaler.transform(X_flat)
        return torch.tensor(X_scaled.reshape(N, W, F), dtype=torch.float32)

    X_train = normalize(X_train_raw)
    X_val = normalize(X_val_raw)
    X_test_norm = normalize(X_test_normal_raw)
    X_test_anom = normalize(X_test_anom_raw)
    
    return X_train, X_val, X_test_norm, X_test_anom, F