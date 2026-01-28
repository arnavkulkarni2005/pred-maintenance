import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path

# ================= CONFIGURATION =================
# Base path to your dataset
DATASET_PATH = Path('/home/kulkarni/projects/pred_maintenance/dataset')
OUTPUT_DIR = Path('/home/kulkarni/projects/pred_maintenance/plots_correlation')

# How many correlation matrices to generate per class
SAMPLES_PER_CLASS = 5

# The specific sensors to check for correlation
TARGET_VARS = ['P-PDG', 'P-TPT', 'T-TPT']

CLASS_NAMES = {
    0: "Normal Operation",
    1: "Abrupt Increase of BSW",
    2: "Spurious Closure of DHSV",
    3: "Severe Slugging",
    4: "Flow Instability",
    5: "Rapid Productivity Loss",
    6: "Rapid Restriction in PCK",
    7: "Scaling in PCK",
    8: "Hydrate Formation"
}
# =================================================

def save_correlation_heatmap(df, filename, class_id):
    """
    Generates and saves a correlation heatmap for the selected sensors.
    """
    # 1. Select available columns
    available_cols = [c for c in TARGET_VARS if c in df.columns]
    
    # We need at least 2 variables to calculate a correlation
    if len(available_cols) < 2:
        return

    # 2. Compute Correlation Matrix
    # method='pearson' is standard. handles NaNs automatically.
    corr_matrix = df[available_cols].corr()

    # 3. Setup the Plot
    plt.figure(figsize=(8, 6))
    
    # Draw Heatmap
    # vmin=-1, vmax=1 ensure the colors are standardized (Red=-1, Green=+1)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdYlGn', 
                vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
    
    class_name = CLASS_NAMES.get(int(class_id), f"Class {class_id}")
    plt.title(f"Sensor Correlation: {class_name}\nFile: {filename}")
    
    # 4. Save Logic
    # Create subfolder: plots_correlation/Class_X
    class_output_dir = OUTPUT_DIR / f"Class_{class_id}"
    class_output_dir.mkdir(parents=True, exist_ok=True)

    clean_name = Path(filename).stem
    save_path = class_output_dir / f"corr_{clean_name}.png"
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close() # Close plot to free memory
    print(f"Saved: {save_path}")

def process_dataset():
    if not DATASET_PATH.exists():
        print(f"Error: Directory not found at {DATASET_PATH}")
        return

    # Create main output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sort folders (0, 1, 2...)
    class_folders = sorted([f for f in DATASET_PATH.iterdir() if f.is_dir()], key=lambda x: x.name)
    
    print(f"Found {len(class_folders)} class folders. Saving to: {OUTPUT_DIR}")

    for class_folder in class_folders:
        try:
            class_id = int(class_folder.name)
        except ValueError:
            continue 

        # Find .parquet files recursively (in REAL/SIMULATED subfolders)
        all_files = list(class_folder.rglob("*.parquet"))
        
        if not all_files:
            continue

        # Random Sample
        n_samples = min(len(all_files), SAMPLES_PER_CLASS)
        selected_files = random.sample(all_files, n_samples)

        print(f"--- Processing Class {class_id} ({n_samples} files) ---")

        for file_path in selected_files:
            try:
                # Load Data
                df = pd.read_parquet(file_path)
                
                # Generate Plot
                save_correlation_heatmap(df, file_path.name, class_id)
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

if __name__ == "__main__":
    process_dataset()