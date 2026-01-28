import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt
from pathlib import Path

# ================= CONFIGURATION =================
DATASET_PATH = Path('/home/kulkarni/projects/pred_maintenance/dataset')
OUTPUT_DIR = Path('/home/kulkarni/projects/pred_maintenance/outlook')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Selected classes (Missing 1, 7, 8)
TARGET_CLASSES = [0, 2, 3, 4, 5, 6] 
SAMPLES_PER_CLASS = 100 
# =================================================

def extract_features(df):
    """Turns a time-series file into a single row of features."""
    features = {}
    
    # Filter for sensor columns only
    cols = [c for c in df.columns if ('P-' in c or 'T-' in c or 'Q' in c) 
            and 'ESTADO' not in c and 'state' not in c.lower()]
    
    for col in cols:
        # Simple stats
        if df[col].notna().any():
            features[f"{col}_mean"] = df[col].mean()
            features[f"{col}_std"] = df[col].std()
            features[f"{col}_min"] = df[col].min()
            features[f"{col}_max"] = df[col].max()
        else:
            # Handle dead sensors with -1 or 0
            features[f"{col}_mean"] = 0
            features[f"{col}_std"] = 0
            features[f"{col}_min"] = 0
            features[f"{col}_max"] = 0
        
    return features

def build_baseline_dataset():
    print("Building Baseline Feature Set...")
    data = []
    labels = []
    
    for class_id in TARGET_CLASSES:
        class_path = DATASET_PATH / str(class_id)
        # Robust file finding
        files = list(class_path.rglob("*.parquet"))[:SAMPLES_PER_CLASS]
        
        for f in files:
            try:
                df = pd.read_parquet(f)
                feat = extract_features(df)
                data.append(feat)
                labels.append(class_id)
            except:
                continue
                
    return pd.DataFrame(data).fillna(0), np.array(labels)

if __name__ == "__main__":
    # 1. Prepare Data
    X, y = build_baseline_dataset()
    
    # --- THE FIX: Encode Labels to 0..N-1 ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Mapped Classes: {le.classes_} -> {np.unique(y_encoded)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded)
    
    print(f"Training XGBoost on {X_train.shape[0]} samples...")
    
    # 2. Train XGBoost
    model = xgb.XGBClassifier(
        objective='multi:softmax', 
        num_class=len(le.classes_), # Automatically set correct number
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    model.fit(X_train, y_train)
    
    # 3. Evaluate
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average='weighted')
    
    print("\n" + "="*30)
    print(f"BASELINE F1 SCORE: {f1:.4f}")
    print("="*30)
    
    # Inverse transform labels for readable report
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=[str(c) for c in le.classes_]))
    
    # 4. Feature Importance Plot
    plt.figure(figsize=(10, 8)) # Taller plot
    xgb.plot_importance(model, max_num_features=15, height=0.5)
    plt.title("Top 15 Features (Statistical Baseline)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "xgboost_feature_importance.png")
    print(f"Saved Feature Importance to {OUTPUT_DIR / 'xgboost_feature_importance.png'}")