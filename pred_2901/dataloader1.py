import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler # <-- Updated Scaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge 

import config

def load_data():
    print("="*50)
    print("🚀 STARTING DATA LOADING PIPELINE (DIAGNOSTIC MODE)")
    print("="*50)
    
    normal_path = config.DATASET_PATH / '0'
    if not normal_path.exists():
        raise FileNotFoundError(f"❌ Folder not found: {normal_path}")
        
    normal_files = list(normal_path.rglob("*.parquet"))
    if config.MAX_FILES_PER_CLASS:
        normal_files = normal_files[:config.MAX_FILES_PER_CLASS * 2] 
    
    train_files, test_files = train_test_split(normal_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42) 

    # --- FIT IMPUTER ---
    print("🧠 FITTING ITERATIVE IMPUTER (BayesianRidge)...")
    train_sample_dfs = []
    sample_size = min(50, len(train_files))
    
    for f in train_files[:sample_size]: 
        temp_df = pd.read_parquet(f)
        for sensor in config.TARGET_SENSORS:
            if sensor not in temp_df.columns:
                temp_df[sensor] = np.nan
        train_sample_dfs.append(temp_df[config.TARGET_SENSORS])
        
    train_concat = pd.concat(train_sample_dfs, ignore_index=True)
    
    for col in train_concat.columns:
        if train_concat[col].isnull().all():
            train_concat[col] = 0.0
            
    global_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42, verbose=0)
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        global_imputer.fit(train_concat)
    print("✅ Imputer fitting complete!")

    def process_files(file_list, desc):
        windows = []
        for f in file_list:
            try:
                df = pd.read_parquet(f)
                for sensor in config.TARGET_SENSORS:
                    if sensor not in df.columns:
                        df[sensor] = np.nan
                df = df[config.TARGET_SENSORS]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    imputed_vals = global_imputer.transform(df) 
                
                df_clean = pd.DataFrame(imputed_vals, columns=config.TARGET_SENSORS)
                df_clean = df_clean.ffill().fillna(0.0) 
                
                # 🛡️ SHIELD 1: Prevent float32 infinity overflow
                vals = np.clip(df_clean.values, -1e9, 1e9) 
                
                if len(vals) < config.WINDOW_SIZE:
                    continue

                for i in range(0, len(vals) - config.WINDOW_SIZE, config.STRIDE):
                    windows.append(vals[i:i+config.WINDOW_SIZE])
                    
            except Exception:
                continue
                
        print(f"🏁 {desc}: Generated {len(windows)} windows.")
        return np.array(windows, dtype=np.float32) if windows else np.array([])

    X_train_raw = process_files(train_files, "Train")
    X_val_raw = process_files(val_files, "Val")
    X_test_normal_raw = process_files(test_files, "Test(Normal)")
    
    # --- PROCESS ANOMALIES INTO A DICTIONARY ---
    print("\n🚨 PROCESSING ANOMALY CLASSES...")
    anomaly_windows_dict = {}
    
    for cls in [1, 2, 3, 4, 5, 6, 7, 8]:
        cls_path = config.DATASET_PATH / str(cls)
        files = list(cls_path.rglob("*.parquet"))
        if config.MAX_FILES_PER_CLASS:
            files = files[:config.MAX_FILES_PER_CLASS]
            
        if files:
            X_anom = process_files(files, f"Class {cls}")
            if len(X_anom) > 0:
                anomaly_windows_dict[cls] = X_anom

    # --- NORMALIZE (RobustScaler + Post-Clipping) ---
    print("\n📏 NORMALIZING DATA (RobustScaler)...")
    N, W, F = X_train_raw.shape
    scaler = RobustScaler()
    scaler.fit(X_train_raw.reshape(-1, F))
    
    def normalize(X):
        if len(X) == 0: return torch.tensor([])
        N, W, F = X.shape
        # Transform using Interquartile Range
        X_scaled = scaler.transform(X.reshape(-1, F))
        
        # 🛡️ THE TRUTH SERUM: Post-Scale Clipping
        # Caps values at 15 IQRs away from the median. 
        # Prevents broken hardware readings from generating Trillion-Level MSEs.
        X_scaled = np.clip(X_scaled, -15.0, 15.0)
        
        return torch.tensor(X_scaled.reshape(N, W, F), dtype=torch.float32)

    X_train = normalize(X_train_raw)
    X_val = normalize(X_val_raw)
    X_test_norm = normalize(X_test_normal_raw)
    
    X_test_anom_dict = {cls: normalize(tensor) for cls, tensor in anomaly_windows_dict.items()}
    
    return X_train, X_val, X_test_norm, X_test_anom_dict, F