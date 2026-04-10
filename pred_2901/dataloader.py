import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge # <-- Updated to fix matrix warnings

# Import centralized configuration
import config

def load_data():
    print("="*50)
    print("🚀 STARTING DATA LOADING PIPELINE")
    print("="*50)
    print(f"📂 Target Dataset Path: {config.DATASET_PATH}")
    print(f"🔧 Target Sensors ({len(config.TARGET_SENSORS)}): {config.TARGET_SENSORS}")
    
    if config.DEBUG_MODE:
        print("⚠️ WARNING: DEBUG_MODE IS ON. Loading artificially limited dataset.")
    
    # 1. Load Normal Data (Class 0)
    normal_path = config.DATASET_PATH / '0'
    print(f"🔍 Searching for Normal data in: {normal_path}")
    if not normal_path.exists():
        raise FileNotFoundError(f"❌ Folder not found: {normal_path}")
        
    normal_files = list(normal_path.rglob("*.parquet"))
    
    if config.MAX_FILES_PER_CLASS:
        normal_files = normal_files[:config.MAX_FILES_PER_CLASS * 2] 
        print(f"✂️ DEBUG_MODE: Trimmed Normal files to {len(normal_files)}")
        
    print(f"✅ Found {len(normal_files)} Normal files.")
    
    if len(normal_files) == 0:
        raise ValueError("❌ CRITICAL: No files found in Class 0!")

    # Split FILES
    train_files, test_files = train_test_split(normal_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42) 
    print(f"📊 Data Split (Files) -> Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    # ==========================================
    # ANTI-LEAKAGE: FIT IMPUTER ON TRAIN ONLY
    # ==========================================
    print("-" * 50)
    print("🧠 PHASE 1: FITTING ITERATIVE IMPUTER (BayesianRidge)")
    train_sample_dfs = []
    
    sample_size = min(50, len(train_files))
    print(f"📥 Loading {sample_size} training files to learn physical correlations...")
    
    for f in train_files[:sample_size]: 
        temp_df = pd.read_parquet(f)
        for sensor in config.TARGET_SENSORS:
            if sensor not in temp_df.columns:
                temp_df[sensor] = np.nan
        train_sample_dfs.append(temp_df[config.TARGET_SENSORS])
        
    train_concat = pd.concat(train_sample_dfs, ignore_index=True)
    print(f"📈 Combined training sample shape for imputer: {train_concat.shape}")
    
    # SAFETY NET for perfectly empty columns
    for col in train_concat.columns:
        if train_concat[col].isnull().all():
            print(f"⚠️ WARNING: Sensor '{col}' is completely missing in training sample. Initializing with 0.0.")
            train_concat[col] = 0.0
    
    print("⚙️ Fitting BayesianRidge Imputer (this might take a moment)...")
    global_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42, verbose=0)
    global_imputer.fit(train_concat)
    print("✅ Imputer fitting complete!")
    print("-" * 50)
    # ==========================================
    
    def process_files(file_list, desc):
        print(f"\n🔄 PROCESSING GROUP: {desc} ({len(file_list)} files)")
        windows = []
        skipped_short = 0
        skipped_error = 0
        
        for idx, f in enumerate(file_list):
            try:
                df = pd.read_parquet(f)
                
                # Check missing sensors for debugging
                missing = [s for s in config.TARGET_SENSORS if s not in df.columns]
                if missing and config.DEBUG_MODE:
                    print(f"  [File {idx}] Missing sensors: {missing}. Queuing for imputation.")
                
                for sensor in config.TARGET_SENSORS:
                    if sensor not in df.columns:
                        df[sensor] = np.nan
                        
                df = df[config.TARGET_SENSORS]

                # 2. TRANSFORM
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    imputed_vals = global_imputer.transform(df) 
                
                df_clean = pd.DataFrame(imputed_vals, columns=config.TARGET_SENSORS)

                # 3. Fill and clean
                df_clean = df_clean.ffill().fillna(0.0) 
                vals = df_clean.values
                vals = np.clip(vals, -1e9, 1e9) 
                
                # Length check
                if len(vals) < config.WINDOW_SIZE:
                    print(f"  ⏭️ SKIPPED '{f.name}': Length ({len(vals)}) is less than window size ({config.WINDOW_SIZE}).")
                    skipped_short += 1
                    continue

                # Window extraction
                extracted_count = 0
                for i in range(0, len(vals) - config.WINDOW_SIZE, config.STRIDE):
                    windows.append(vals[i:i+config.WINDOW_SIZE])
                    extracted_count += 1
                    
                # print(f"  ✔️ SUCCESS '{f.name}': Extracted {extracted_count} windows.") # Uncomment if you want extreme verbosity
                    
            except Exception as e:
                print(f"  ❌ ERROR SKIPPING '{f.name}': {str(e)}")
                skipped_error += 1
                continue
                
        print(f"🏁 DONE {desc}: Generated {len(windows)} total windows.")
        print(f"   -> Skipped (Too Short): {skipped_short}")
        print(f"   -> Skipped (Errors): {skipped_error}")
        
        return np.array(windows, dtype=np.float32) if windows else np.array([])

    print("\n" + "="*50)
    print("🏭 PHASE 2: PROCESSING DATASETS")
    X_train_raw = process_files(train_files, "Train")
    if len(X_train_raw) == 0:
        raise ValueError("❌ CRITICAL: X_train generated 0 windows. Pipeline cannot continue.")

    X_val_raw = process_files(val_files, "Val")
    X_test_normal_raw = process_files(test_files, "Test(Normal)")
    
    # PROCESS ANOMALY DATA
    print("\n" + "="*50)
    print("🚨 PHASE 3: PROCESSING ANOMALY CLASSES")
    anomaly_windows = []
    
    for cls in [1, 2, 3, 4, 5, 6, 7, 8]:
        cls_path = config.DATASET_PATH / str(cls)
        files = list(cls_path.rglob("*.parquet"))
        
        if config.MAX_FILES_PER_CLASS:
            files = files[:config.MAX_FILES_PER_CLASS]
            
        if files:
            X_anom = process_files(files, f"Class {cls}")
            if len(X_anom) > 0:
                anomaly_windows.append(X_anom)
        else:
            print(f"⚠️ WARNING: No files found for Class {cls} in {cls_path}")
    
    if anomaly_windows:
        X_test_anom_raw = np.concatenate(anomaly_windows)
        print(f"\n✅ Total Anomaly Windows Extracted: {len(X_test_anom_raw)}")
    else:
        X_test_anom_raw = np.array([])
        print("\n❌ CRITICAL WARNING: 0 Anomaly windows extracted across all classes.")

    # Normalize
    print("\n" + "="*50)
    print("📏 PHASE 4: NORMALIZATION (StandardScaler)")
    N, W, F = X_train_raw.shape
    print(f"Fitting Scaler on Training Tensor: {N} windows, {W} timesteps, {F} features.")
    
    scaler = StandardScaler()
    X_train_flat = X_train_raw.reshape(-1, F)
    scaler.fit(X_train_flat)
    print("✅ Scaler fitted successfully.")
    
    def normalize(X, name):
        if len(X) == 0: 
            print(f"⚠️ Skipping normalization for {name} (Empty Tensor)")
            return torch.tensor([])
        N, W, F = X.shape
        X_flat = X.reshape(-1, F)
        X_scaled = scaler.transform(X_flat)
        print(f"✅ Normalized {name}: Shape {X.shape}")
        return torch.tensor(X_scaled.reshape(N, W, F), dtype=torch.float32)

    X_train = normalize(X_train_raw, "X_train")
    X_val = normalize(X_val_raw, "X_val")
    X_test_norm = normalize(X_test_normal_raw, "X_test_norm")
    X_test_anom = normalize(X_test_anom_raw, "X_test_anom")
    
    print("\n🎉 DATA LOADING COMPLETE. Handing off to model...")
    print("="*50 + "\n")
    return X_train, X_val, X_test_norm, X_test_anom, F