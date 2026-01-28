import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

base_path = '/home/kulkarni/projects/pred_maintenance/dataset'
output_dir = '/home/kulkarni/projects/pred_maintenance/plots/precise_color_gallery'
os.makedirs(output_dir, exist_ok=True)

def plot_with_exact_labels(label_dir, filename):
    file_path = os.path.join(base_path, label_dir, filename)
    save_path = os.path.join(output_dir, f"class_{label_dir}_{filename.replace('.parquet', '.png')}")
    
    try:
        df = pd.read_parquet(file_path)
        
        # Identify the Label Column
        label_col = next((c for c in df.columns if c.lower() in ['label', 'class', 'target']), None)
        
        if label_col is None:
            return False

        # FIX: Handle missing values in the label column to prevent boolean conversion errors
        # Fill NaNs with -1 so they don't trigger the 0 (Normal) or >0 (Anomaly) logic
        clean_labels = df[label_col].fillna(-1)

        # Select Sensors
        sensors = ['P-PDG', 'P-TPT', 'T-TPT']
        available = [s for s in sensors if s in df.columns]
        if not available: available = df.columns[:3].tolist()

        fig, axes = plt.subplots(len(available), 1, figsize=(15, 12), sharex=True)
        if len(available) == 1: axes = [axes]

        for i, sensor in enumerate(available):
            axes[i].plot(df.index, df[sensor], color='black', linewidth=0.7)
            
            # Precise Time-Step Coloring with NaN handling
            # Green = confirmed Normal (0), Red = confirmed Anomaly (>0)
            axes[i].fill_between(df.index, df[sensor].min(), df[sensor].max(), 
                                 where=(clean_labels == 0), color='green', alpha=0.15, 
                                 transform=axes[i].get_xaxis_transform())
            
            axes[i].fill_between(df.index, df[sensor].min(), df[sensor].max(), 
                                 where=(clean_labels > 0), color='red', alpha=0.15, 
                                 transform=axes[i].get_xaxis_transform())
            
            axes[i].set_ylabel(sensor)
            axes[i].grid(True, alpha=0.3)

        plt.suptitle(f"Precise Mapping - Class {label_dir}: {filename}")
        plt.xlabel("Time Index")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        
        # Memory Management: Explicitly close the figure
        plt.close(fig)
        return True

    except Exception as e:
        print(f"Error on {filename}: {e}")
        return False

# Execute for variety across all folders
folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
for label in folders:
    folder_path = os.path.join(base_path, label)
    files = sorted(os.listdir(folder_path))
    # Select first few files of each source type for variety
    selected = (
        [f for f in files if f.startswith('WELL')][:3] +
        [f for f in files if f.startswith('SIMULATED')][:3] +
        [f for f in files if f.startswith('DRAWN')][:3]
    )
    
    print(f"\nProcessing Class {label}...")
    for f in tqdm(selected):
        plot_with_exact_labels(label, f)