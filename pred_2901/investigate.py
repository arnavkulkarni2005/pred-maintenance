"""
investigate.py

Diagnostic tool: analyzes reconstruction error on a single file
chronologically to verify that the model is detecting anomalies
rather than overfitting or leaking.

How to use
----------
# List all available anomaly files:
    python investigate.py --list

# Investigate first file of anomaly class 3 (auto-loads saved model):
    python investigate.py --class 3

# Investigate a specific file index within a class:
    python investigate.py --class 5 --file-index 2

# Investigate a normal file (class 0) to check false-positive rate:
    python investigate.py --class 0 --file-index 0

# Investigate an arbitrary parquet file:
    python investigate.py --file /path/to/file.parquet

# Override the anomaly threshold (default: auto from first 20% of windows):
    python investigate.py --class 3 --threshold 1.5

# Force a fresh quick-train even if a saved model exists:
    python investigate.py --class 3 --retrain

What the plots show
-------------------
Row 1 — Raw sensor values (first two sensors, unscaled)
Row 2 — Scaled sensor values after imputation + RobustScaler (model input)
Row 3 — Per-window reconstruction MSE over time.
         Red shading above the threshold = windows flagged as anomalies.

A well-calibrated model should show MSE near 0 during the normal
phase and a clear spike when the fault begins.
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import config
from model import TunableAutoencoder
from strict_dataloader import (
    get_file_paths,
    load_preprocessors,
    process_and_fit_preprocessors,
    process_files_to_windows,
    save_preprocessors,
)

warnings.filterwarnings("ignore")


# ==================================================================== #
#  Model loading / quick-training                                       #
# ==================================================================== #

def load_saved_model():
    """
    Attempts to load the final model and preprocessors saved by
    master_pipeline.py from config.OUTPUT_DIR.

    Returns (model, imputer, scaler) on success, raises FileNotFoundError
    if files are missing.
    """
    ckpt_path = config.OUTPUT_DIR / 'final_model.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No saved model at {ckpt_path}")

    print(f"Loading model from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE)

    model = TunableAutoencoder(
        input_dim=ckpt['n_feats'],
        window_size=ckpt['window_size'],
        hidden_layers=ckpt['hidden_layers'],
    ).to(config.DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    imputer, scaler = load_preprocessors(config.OUTPUT_DIR)
    print(f"  {model}")
    print("  Preprocessors loaded.")
    return model, imputer, scaler


def quick_train_model(normal_files, n_epochs=5):
    """
    Trains a small autoencoder for `n_epochs` on all normal files.
    Used as a fallback when no saved model exists, or when --retrain
    is passed.

    Saves the model and preprocessors to config.OUTPUT_DIR so
    subsequent calls can load rather than retrain.

    Returns (model, imputer, scaler).
    """
    print(f"Training quick diagnostic model ({n_epochs} epochs)...")
    imputer, scaler = process_and_fit_preprocessors(normal_files)
    save_preprocessors(imputer, scaler, config.OUTPUT_DIR)

    X_train = process_files_to_windows(normal_files, imputer, scaler)
    n_feats = X_train.shape[2]

    model = TunableAutoencoder(
        input_dim=n_feats,
        window_size=config.WINDOW_SIZE,
        hidden_layers=[128, 64],
    ).to(config.DEVICE)
    print(f"  {model}")

    loader    = DataLoader(TensorDataset(X_train), batch_size=512, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        model.train()
        total = 0.0
        for (x,) in loader:
            x = x.to(config.DEVICE)
            optimizer.zero_grad()
            loss = torch.mean((model(x) - x) ** 2)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"  Epoch {epoch + 1}/{n_epochs}  loss={total / len(loader):.6f}")

    # Save so next run doesn't retrain
    ckpt_path = config.OUTPUT_DIR / 'final_model.pt'
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'best_params':      {'n_layers': 2, 'layer_0_dim': 128,
                                 'layer_1_dim': 64, 'dropout': 0.0, 'lr': 1e-3},
            'n_feats':          n_feats,
            'hidden_layers':    [128, 64],
            'window_size':      config.WINDOW_SIZE,
        },
        ckpt_path,
    )
    print(f"  Quick model saved → {ckpt_path}")

    model.eval()
    return model, imputer, scaler


def get_model(normal_files, force_retrain=False):
    """
    Returns (model, imputer, scaler).
    Loads saved model if available; falls back to quick training.
    """
    if force_retrain:
        print("--retrain flag set: training fresh model.")
        return quick_train_model(normal_files)

    try:
        return load_saved_model()
    except FileNotFoundError as e:
        print(f"  {e}")
        print("  No saved model found — running quick training (5 epochs).")
        print("  For better results, run master_pipeline.py first.")
        return quick_train_model(normal_files)


# ==================================================================== #
#  Core analysis                                                        #
# ==================================================================== #

def analyze_file(model, file_path, imputer, scaler, threshold=None):
    """
    Runs the model chronologically over a single parquet file.

    Parameters
    ----------
    model     : trained TunableAutoencoder (eval mode)
    file_path : Path or str
    imputer   : fitted IterativeImputer
    scaler    : fitted RobustScaler
    threshold : float or None
        MSE level above which a window is flagged as anomalous.
        If None, auto-computes from mean + 3 std of the first 20%
        of windows (assumed to be normal / pre-fault).

    Saves a 3-panel diagnostic plot to config.OUTPUT_DIR and prints
    a summary of detected anomaly windows.

    Returns
    -------
    losses : np.ndarray of shape [n_windows]
    """
    file_path = Path(file_path)
    print(f"\nAnalyzing: {file_path.name}  ({file_path.parent.name})")

    # ---- 1. Load and preprocess ------------------------------------ #
    df = pd.read_parquet(file_path)
    for sensor in config.TARGET_SENSORS:
        if sensor not in df.columns:
            df[sensor] = np.nan
    df = df[config.TARGET_SENSORS]

    raw = df.values.astype(np.float32)
    raw = np.where(np.isfinite(raw), raw, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imputed = imputer.transform(raw)

    scaled = scaler.transform(imputed)
    clean  = np.clip(scaled, -config.CLIP_VALUE, config.CLIP_VALUE)

    # ---- 2. Build chronological (non-shuffled) windows ------------- #
    windows     = []
    end_indices = []           # timestep at END of each window
    for i in range(0, len(clean) - config.WINDOW_SIZE + 1, config.STRIDE):
        windows.append(clean[i: i + config.WINDOW_SIZE])
        end_indices.append(i + config.WINDOW_SIZE)

    if not windows:
        print(f"  File has only {len(clean)} timesteps — "
              f"need at least {config.WINDOW_SIZE}. Skipping.")
        return None

    X = torch.tensor(np.array(windows, dtype=np.float32)).to(config.DEVICE)

    # ---- 3. Compute per-window MSE --------------------------------- #
    model.eval()
    with torch.no_grad():
        losses = model.reconstruction_error(X).cpu().numpy()

    time_axis = np.array(end_indices)

    # ---- 4. Threshold ---------------------------------------------- #
    if threshold is None:
        n_ref     = max(10, int(0.2 * len(losses)))
        ref       = losses[:n_ref]
        threshold = float(ref.mean() + 3.0 * ref.std())
        print(f"  Auto-threshold (mean + 3σ of first 20% = {n_ref} windows): "
              f"{threshold:.4f}")
    else:
        print(f"  Manual threshold: {threshold:.4f}")

    anomaly_mask = losses > threshold

    # ---- 5. Plot --------------------------------------------------- #
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=False)
    fig.suptitle(
        f'Anomaly Investigation — {file_path.name}  '
        f'[class {file_path.parent.name}]',
        fontsize=13,
    )

    # Panel 1: raw sensor values (first 2 sensors)
    ax1 = axes[0]
    for sensor in config.TARGET_SENSORS[:2]:
        ax1.plot(df[sensor].values, linewidth=0.8, alpha=0.85, label=sensor)
    ax1.set_title('Raw sensor readings (unscaled)')
    ax1.set_ylabel('Raw value')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(alpha=0.3)

    # Panel 2: scaled values (what the model sees)
    ax2 = axes[1]
    for i, sensor in enumerate(config.TARGET_SENSORS[:2]):
        ax2.plot(clean[:, i], linewidth=0.8, alpha=0.85, label=sensor)
    ax2.axhline( config.CLIP_VALUE, color='grey', linestyle=':', linewidth=0.7, alpha=0.6)
    ax2.axhline(-config.CLIP_VALUE, color='grey', linestyle=':', linewidth=0.7, alpha=0.6)
    ax2.set_title(f'Scaled values (model input, clipped ±{config.CLIP_VALUE})')
    ax2.set_ylabel('Scaled value')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(alpha=0.3)

    # Panel 3: reconstruction error
    ax3 = axes[2]
    ax3.fill_between(time_axis, losses, alpha=0.25, color='tomato')
    ax3.plot(time_axis, losses, color='tomato', linewidth=1.2, label='MSE')

    if anomaly_mask.any():
        ax3.fill_between(
            time_axis, 0, losses,
            where=anomaly_mask,
            color='darkred', alpha=0.55,
            label=f'Anomaly (>{threshold:.3f})',
        )

    ax3.axhline(
        threshold,
        color='black', linestyle='--', linewidth=1.0,
        label=f'Threshold = {threshold:.3f}',
    )
    ax3.set_title('Reconstruction error (MSE) — anomaly signal')
    ax3.set_ylabel('MSE')
    ax3.set_xlabel('Timestep')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(alpha=0.3)

    plt.tight_layout()

    out_name = f'investigate_{file_path.parent.name}_{file_path.stem}.png'
    out_path = config.OUTPUT_DIR / out_name
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved → {out_path}")

    # ---- 6. Summary ------------------------------------------------ #
    n_anom = int(anomaly_mask.sum())
    pct    = 100.0 * n_anom / len(losses)
    print(
        f"  Windows: {len(losses)}  |  "
        f"Anomaly: {n_anom} ({pct:.1f}%)  |  "
        f"MSE — min={losses.min():.4f}  "
        f"mean={losses.mean():.4f}  "
        f"max={losses.max():.4f}"
    )

    return losses


# ==================================================================== #
#  Helpers                                                              #
# ==================================================================== #

def list_files(normal_files, anomaly_files_dict):
    """Prints a summary table of available files."""
    print("\nAvailable files")
    print(f"  {'Class':<8}  {'Count':>6}  First file")
    print("  " + "-" * 56)
    print(f"  {'0 (normal)':<8}  {len(normal_files):>6}  "
          f"{Path(normal_files[0]).name if normal_files else '—'}")
    for cls in sorted(anomaly_files_dict.keys()):
        files = anomaly_files_dict[cls]
        name  = Path(files[0]).name if files else '—'
        print(f"  {cls:<8}  {len(files):>6}  {name}")
    print()


# ==================================================================== #
#  Entry point                                                          #
# ==================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Investigate reconstruction error on a single file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--class', type=int, dest='anomaly_class', default=None,
        help='Class directory to pick a file from (0=normal, 1-8=anomaly).',
    )
    parser.add_argument(
        '--file-index', type=int, default=0,
        help='Which file within the class to use (default: 0).',
    )
    parser.add_argument(
        '--file', type=str, default=None,
        help='Direct path to a specific .parquet file.',
    )
    parser.add_argument(
        '--threshold', type=float, default=None,
        help='Manual MSE anomaly threshold (auto-computed if not set).',
    )
    parser.add_argument(
        '--retrain', action='store_true',
        help='Force a fresh quick-train even if a saved model exists.',
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List all available files and exit.',
    )
    args = parser.parse_args()

    # Discover all files
    normal_files, anomaly_files_dict = get_file_paths()

    if args.list:
        list_files(normal_files, anomaly_files_dict)
        raise SystemExit(0)

    # Resolve which file to analyze
    if args.file:
        target_file = Path(args.file)
        if not target_file.exists():
            raise FileNotFoundError(f"File not found: {target_file}")

    elif args.anomaly_class is not None:
        cls = args.anomaly_class

        if cls == 0:
            file_pool = normal_files
        elif cls in anomaly_files_dict:
            file_pool = anomaly_files_dict[cls]
        else:
            available = sorted(anomaly_files_dict.keys())
            raise ValueError(
                f"Class {cls} not found. Available anomaly classes: {available}"
            )

        if args.file_index >= len(file_pool):
            raise IndexError(
                f"--file-index {args.file_index} is out of range; "
                f"class {cls} has {len(file_pool)} files."
            )

        target_file = file_pool[args.file_index]

    else:
        # Default: first file of the first anomaly class
        first_cls   = sorted(anomaly_files_dict.keys())[0]
        target_file = anomaly_files_dict[first_cls][0]
        print(f"No target specified — defaulting to class {first_cls}, file 0.")

    # Load or train model
    model, imputer, scaler = get_model(normal_files, force_retrain=args.retrain)

    # Run analysis
    analyze_file(model, target_file, imputer, scaler, threshold=args.threshold)