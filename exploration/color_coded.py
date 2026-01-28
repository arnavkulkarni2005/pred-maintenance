import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from pathlib import Path

# ================= CONFIGURATION =================
# Input dataset path
DATASET_PATH = Path("/home/kulkarni/projects/pred_maintenance/dataset")

# Output folder for images
OUTPUT_DIR = Path("new_plots")

# Number of plots to generate per class
SAMPLES_PER_CLASS = 10 

# The specific variables to plot
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

def save_colored_plot(df, original_filename, class_id):
    """
    Generates and saves the plot to the output directory.
    """
    # Check for available variables
    available_vars = [var for var in TARGET_VARS if var in df.columns]
    
    if not available_vars:
        return # Skip if data is missing

    n_vars = len(available_vars)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3*n_vars), sharex=True)
    if n_vars == 1: axes = [axes]

    class_name = CLASS_NAMES.get(int(class_id), f"Class {class_id}")

    for i, col in enumerate(available_vars):
        ax = axes[i]
        
        # 1. Base Layer: Anomaly (Red)
        ax.plot(df.index, df[col], color='#ff4d4d', label='Anomaly', linewidth=2)
        
        # 2. Overlay Layer: Normal (Green)
        normal_segment = df[col].copy()
        class_col = 'class' if 'class' in df.columns else 'class_id'
        
        if class_col in df.columns:
            # Mask fault values (make them NaN)
            normal_segment[df[class_col] != 0] = np.nan
        
        ax.plot(df.index, normal_segment, color='#2ecc71', label='Normal', linewidth=2)

        ax.set_ylabel(col)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        if i == 0:
            ax.legend(loc='upper right', framealpha=0.9)

    plt.suptitle(f"{class_name}\nFile: {original_filename}", y=1.005, fontsize=14, fontweight='bold')
    plt.xlabel("Time")
    plt.tight_layout()

    # --- SAVE LOGIC ---
    # Create a subfolder for the class (e.g., new_plots/Class_1)
    class_output_dir = OUTPUT_DIR / f"Class_{class_id}"
    class_output_dir.mkdir(parents=True, exist_ok=True)

    # Clean the filename for saving (remove .parquet extension)
    clean_name = Path(original_filename).stem
    save_path = class_output_dir / f"{clean_name}.png"
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig) # Important: Close memory to prevent crash
    print(f"Saved: {save_path}")

def process_dataset():
    if not DATASET_PATH.exists():
        print(f"Error: Directory not found at {DATASET_PATH}")
        return

    # Create the main output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_folders = sorted([f for f in DATASET_PATH.iterdir() if f.is_dir()], key=lambda x: x.name)
    print(f"Found {len(class_folders)} class folders. Saving to: {OUTPUT_DIR.resolve()}")

    for class_folder in class_folders:
        try:
            class_id = int(class_folder.name)
        except ValueError:
            continue 

        all_files = list(class_folder.rglob("*.parquet"))
        
        if not all_files:
            continue

        n_samples = min(len(all_files), SAMPLES_PER_CLASS)
        selected_files = random.sample(all_files, n_samples)

        print(f"--- Processing Class {class_id} ({n_samples} files) ---")

        for file_path in selected_files:
            try:
                df = pd.read_parquet(file_path)
                
                # Ensure datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.set_index('timestamp')
                
                save_colored_plot(df, file_path.name, class_id)
                
            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")

if __name__ == "__main__":
    process_dataset()