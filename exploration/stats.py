import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.fft import fft, fftfreq

# ================= CONFIGURATION =================
DATASET_PATH = Path('/home/kulkarni/projects/pred_maintenance/dataset')
OUTPUT_DIR = Path('/home/kulkarni/projects/pred_maintenance/presentation_plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = [0, 2, 3] 
SENSOR = 'P-PDG' 
# =================================================

def analyze_dataset_balance():
    """Step 2: Count examples per class."""
    counts = []
    # Sort class folders numerically
    class_folders = sorted([f for f in DATASET_PATH.iterdir() if f.is_dir()], key=lambda x: int(x.name) if x.name.isdigit() else 999)
    
    for f in class_folders:
        if not f.name.isdigit(): continue
        class_id = int(f.name)
        n_files = len(list(f.rglob("*.parquet")))
        counts.append({'Class': class_id, 'Count': n_files})
            
    df_counts = pd.DataFrame(counts)
    
    plt.figure(figsize=(10, 5))
    # FIX: Added hue=Class and legend=False to silence warning
    sns.barplot(data=df_counts, x='Class', y='Count', hue='Class', palette='viridis', legend=False)
    plt.title("Step 2: Dataset Balance (Number of Sequences per Class)")
    plt.ylabel("Number of Files")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(OUTPUT_DIR / "step2_class_balance.png")
    plt.close()
    print("Saved Step 2: Class Balance Plot")

def analyze_statistics():
    """Step 6: Statistics Comparison (with filter for bad files)."""
    stats_data = []

    for class_id in TARGET_CLASSES:
        class_path = DATASET_PATH / str(class_id)
        files = list(class_path.rglob("*.parquet"))[:40] # Check more files to find good ones
        
        for file in files:
            try:
                df = pd.read_parquet(file)
                if SENSOR in df.columns:
                    # FILTER: Skip files where sensor is dead (mean < 10 bar)
                    # Real pressure is usually > 100 bar (1e7 Pa)
                    if df[SENSOR].mean() < 1e5: 
                        continue

                    stats_data.append({
                        'Class': str(class_id),
                        'Mean': df[SENSOR].mean(),
                        'Std Dev': df[SENSOR].std()
                    })
            except:
                continue

    df_stats = pd.DataFrame(stats_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # FIX: Added hue='Class' and legend=False
    sns.boxplot(data=df_stats, x='Class', y='Mean', hue='Class', ax=axes[0], palette='Set2', legend=False)
    axes[0].set_title(f"Difference in Mean {SENSOR}")
    
    sns.boxplot(data=df_stats, x='Class', y='Std Dev', hue='Class', ax=axes[1], palette='Set2', legend=False)
    axes[1].set_title(f"Difference in Variance {SENSOR}")
    
    plt.suptitle("Step 6: Statistical Differences (Normal vs Anomalies)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step6_stats_comparison.png")
    plt.close()
    print("Saved Step 6: Statistics Comparison Plot")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq

def analyze_fft_frequencies(max_freq=0.02):
    """
    Step 5: FFT Analysis (Detrended).
    
    Args:
        max_freq (float): The upper limit for the x-axis in Hz. 
                          Set to None to show the full spectrum.
    """
    fig, axes = plt.subplots(len(TARGET_CLASSES), 1, figsize=(10, 10), sharex=True)
    
    # Handle case where there is only one class (axes is not a list)
    if len(TARGET_CLASSES) == 1:
        axes = [axes]

    for i, class_id in enumerate(TARGET_CLASSES):
        class_path = DATASET_PATH / str(class_id)
        files = list(class_path.rglob("*.parquet"))
        
        # Find a valid file (non-empty, non-zero)
        valid_df = None
        for f in files:
            try:
                temp_df = pd.read_parquet(f)
                # Check if sensor exists and has realistic pressure (> 10 bar)
                if SENSOR in temp_df.columns and temp_df[SENSOR].mean() > 1e6:
                    valid_df = temp_df
                    break
            except:
                continue
        
        if valid_df is None:
            print(f"No valid file found for Class {class_id}")
            continue
        
        # FIX: Updated fillna syntax
        raw_signal = valid_df[SENSOR].ffill().bfill().values
        
        # FIX: DETRENDING (Subtract Mean)
        signal = raw_signal - np.mean(raw_signal)
        
        # Perform FFT
        N = len(signal)
        
        # NOTE: Ensure the second argument (d=1) matches your data's sampling interval.
        # If your data is not 1Hz, change '1' to '1/sampling_rate'.
        yf = fft(signal)
        xf = fftfreq(N, 1)[:N//2] 
        amplitude = 2.0/N * np.abs(yf[0:N//2])
        
        ax = axes[i]
        ax.plot(xf, amplitude, color='purple')
        ax.set_title(f"Frequency Spectrum (FFT) - Class {class_id} ({SENSOR})")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.5)
        
        # --- MODIFICATION HERE ---
        # Set the x-axis limit based on the argument
        if max_freq:
            ax.set_xlim(0, max_freq)
        else:
            # If max_freq is None, matplotlib will autoscaling to the full range
            ax.autoscale(enable=True, axis='x', tight=True) 
        # -------------------------

    plt.xlabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step5_fft_analysis.png")
    plt.close()
    print(f"Saved Step 5: FFT Frequency Analysis (0 - {max_freq if max_freq else 'Max'} Hz)")

if __name__ == "__main__":
    analyze_dataset_balance()
    analyze_statistics()
    analyze_fft_frequencies()