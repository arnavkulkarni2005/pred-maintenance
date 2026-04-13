"""
master_pipeline.py

Full end-to-end anomaly detection pipeline. Run this to train and
evaluate the system. Outputs are saved to config.OUTPUT_DIR.

Pipeline stages
---------------
Phase 1 — Optuna hyperparameter tuning
    Finds the best MLP architecture (layer sizes, dropout, lr) by
    minimising normal reconstruction MSE on a held-out validation set.
    Uses a MedianPruner to kill poor trials early.

Phase 2 — Strict 4-fold cross-validation
    Evaluates generalisation using FILE-level folds so no recording
    from a val file can appear in training. Preprocessors (imputer +
    scaler) are re-fitted from scratch inside each fold — zero leakage.
    Saves per-fold ROC plots and a final AUROC summary CSV.

Phase 3 — Final model training
    Trains one model on ALL normal files using the best hyperparameters.
    Saves the model weights and preprocessors so investigate.py can
    load them without retraining.

Outputs (all in config.OUTPUT_DIR)
-----------------------------------
  optuna_trials.csv          — full Optuna trial history
  best_params.csv            — best hyperparameter set
  fold_{n}_roc.png           — ROC curves for each fold
  kfold_summary.csv          — mean / max / std AUROC per anomaly class
  final_model.pt             — saved final model checkpoint
  imputer.joblib             — fitted IterativeImputer
  scaler.joblib              — fitted RobustScaler

Usage
-----
    python master_pipeline.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import optuna
from optuna.pruners import MedianPruner
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

import config
from model import TunableAutoencoder
from strict_dataloader import (
    get_file_paths,
    process_and_fit_preprocessors,
    process_files_to_windows,
    save_preprocessors,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==================================================================== #
#  Training helper                                                      #
# ==================================================================== #

def _train_model(model, train_loader, n_epochs, lr):
    """
    Trains `model` in-place for `n_epochs` using Adam.
    Prints a loss line every 5 epochs.
    Returns a list of per-epoch mean losses.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epoch_losses = []

    for epoch in range(n_epochs):
        model.train()
        batch_losses = []
        for (x,) in train_loader:
            x = x.to(config.DEVICE)
            optimizer.zero_grad()
            loss = torch.mean((model(x) - x) ** 2)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        mean_loss = float(np.mean(batch_losses))
        epoch_losses.append(mean_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == n_epochs - 1:
            print(f"    Epoch {epoch + 1:3d}/{n_epochs}  loss={mean_loss:.6f}")

    return epoch_losses


# ==================================================================== #
#  Evaluation helper                                                    #
# ==================================================================== #

def _evaluate_fold(model, val_loader, anom_tensors, fold_idx):
    """
    Computes per-class AUROC for one fold.
    Saves a ROC curve plot to config.OUTPUT_DIR.
    Returns {class_int: auroc_float}.
    """
    model.eval()

    # Normal reconstruction losses
    norm_losses = []
    with torch.no_grad():
        for (x,) in val_loader:
            x = x.to(config.DEVICE)
            norm_losses.extend(model.reconstruction_error(x).cpu().numpy())
    norm_losses = np.array(norm_losses)

    class_aurocs = {}
    fig, ax = plt.subplots(figsize=(10, 8))

    for cls, anom_tensor in sorted(anom_tensors.items()):
        if len(anom_tensor) == 0:
            continue

        anom_losses = []
        anom_loader = DataLoader(TensorDataset(anom_tensor), batch_size=config.BATCH_SIZE)
        with torch.no_grad():
            for (x,) in anom_loader:
                x = x.to(config.DEVICE)
                anom_losses.extend(model.reconstruction_error(x).cpu().numpy())
        anom_losses = np.array(anom_losses)

        y_true  = np.concatenate([np.zeros(len(norm_losses)), np.ones(len(anom_losses))])
        y_score = np.concatenate([norm_losses, anom_losses])

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        class_aurocs[cls] = roc_auc

        ax.plot(fpr, tpr, label=f'Class {cls}  AUC={roc_auc:.3f}')
        print(f"    Class {cls} AUROC: {roc_auc:.4f}")

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves — Strict File-Split Fold {fold_idx}')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

    out_path = config.OUTPUT_DIR / f'fold_{fold_idx}_roc.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    ROC plot → {out_path}")

    return class_aurocs


# ==================================================================== #
#  Phase 1 — Optuna                                                     #
# ==================================================================== #

def run_optuna_phase(normal_files):
    """
    Splits normal files 80/20, fits preprocessors on the 80% split,
    then runs Optuna to minimise reconstruction MSE on the 20% split.

    Hyperparameters searched
    ------------------------
    n_layers          : 1–3
    layer_{i}_dim     : 16–256 (decreasing, enforces bottleneck)
    dropout           : 0.0–0.3
    lr                : 1e-4 – 5e-3 (log scale)

    Returns
    -------
    best_params : dict
    n_feats     : int  (number of sensor features)
    """
    print("\n" + "=" * 60)
    print("PHASE 1: OPTUNA HYPERPARAMETER TUNING")
    print(f"  Trials: {config.N_TRIALS}  |  Epochs/trial: {config.TRIAL_EPOCHS}")
    print("=" * 60)

    opt_train_files, opt_val_files = train_test_split(
        normal_files, test_size=0.2, random_state=42
    )

    opt_imputer, opt_scaler = process_and_fit_preprocessors(opt_train_files)

    print("  Generating Optuna tensors...")
    X_train_opt = process_files_to_windows(opt_train_files, opt_imputer, opt_scaler)
    X_val_opt   = process_files_to_windows(opt_val_files,   opt_imputer, opt_scaler)
    print(f"  Shapes — train: {X_train_opt.shape}  val: {X_val_opt.shape}")

    n_feats = X_train_opt.shape[2]

    def objective(trial):
        # Build a strictly decreasing hidden_layers list (bottleneck structure)
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_layers = []
        prev_dim = config.WINDOW_SIZE * n_feats

        for i in range(n_layers):
            # Each layer can be at most half the previous dimension
            max_dim = max(16, prev_dim // 2)
            dim = trial.suggest_int(
                f'layer_{i}_dim', 16, min(256, max_dim), step=16
            )
            hidden_layers.append(dim)
            prev_dim = dim

        dropout = trial.suggest_float('dropout', 0.0, 0.3)
        lr      = trial.suggest_float('lr', 1e-4, 5e-3, log=True)

        model = TunableAutoencoder(
            n_feats, config.WINDOW_SIZE, hidden_layers, dropout
        ).to(config.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_loader = DataLoader(
            TensorDataset(X_train_opt), batch_size=config.BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val_opt), batch_size=config.BATCH_SIZE
        )

        for epoch in range(config.TRIAL_EPOCHS):
            model.train()
            for (x,) in train_loader:
                x = x.to(config.DEVICE)
                optimizer.zero_grad()
                loss = torch.mean((model(x) - x) ** 2)
                loss.backward()
                optimizer.step()

            # Validation MSE for pruner
            model.eval()
            val_mse = 0.0
            with torch.no_grad():
                for (x,) in val_loader:
                    x = x.to(config.DEVICE)
                    val_mse += torch.mean((model(x) - x) ** 2).item()
            val_mse /= len(val_loader)

            trial.report(val_mse, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_mse

    pruner = MedianPruner(
        n_startup_trials=config.PRUNER_STARTUP_TRIALS,
        n_warmup_steps=config.PRUNER_WARMUP_STEPS,
    )
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=config.N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    print(f"\n  Best params : {best_params}")
    print(f"  Best val MSE: {study.best_value:.6f}")

    # Persist Optuna history
    study.trials_dataframe().to_csv(
        config.OUTPUT_DIR / 'optuna_trials.csv', index=False
    )

    return best_params, n_feats


# ==================================================================== #
#  Phase 2 — K-Fold cross-validation                                   #
# ==================================================================== #

def run_kfold_phase(normal_files, anomaly_files_dict, best_params, n_feats):
    """
    Runs 4-fold cross-validation at the FILE level.
    Preprocessors are re-fitted inside every fold — no leakage.

    Returns
    -------
    List of {class: auroc} dicts, one per fold.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: STRICT 4-FOLD CROSS-VALIDATION")
    print("=" * 60)

    hidden_layers = [
        best_params[f'layer_{i}_dim'] for i in range(best_params['n_layers'])
    ]
    print(f"  Architecture: {n_feats * config.WINDOW_SIZE} → {hidden_layers} "
          f"→ {n_feats * config.WINDOW_SIZE}")

    normal_arr = np.array(normal_files)
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    all_fold_aurocs = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(normal_arr), start=1):
        print(f"\n--- FOLD {fold_idx} / 4 ---")

        fold_train_files = normal_arr[train_idx].tolist()
        fold_val_files   = normal_arr[val_idx].tolist()

        # Fit strictly on this fold's training files
        fold_imputer, fold_scaler = process_and_fit_preprocessors(fold_train_files)

        print("  Generating windows...")
        X_train = process_files_to_windows(fold_train_files, fold_imputer, fold_scaler)
        X_val   = process_files_to_windows(fold_val_files,   fold_imputer, fold_scaler)

        # Anomaly windows normalised by THIS fold's scaler
        fold_anom = {}
        for cls, f_list in anomaly_files_dict.items():
            t = process_files_to_windows(f_list, fold_imputer, fold_scaler)
            if len(t) > 0:
                fold_anom[cls] = t.to(config.DEVICE)

        print(f"  Shapes — train: {X_train.shape}  val: {X_val.shape}  "
              f"anomaly classes: {sorted(fold_anom.keys())}")

        model = TunableAutoencoder(
            input_dim=n_feats,
            window_size=config.WINDOW_SIZE,
            hidden_layers=hidden_layers,
            dropout=best_params['dropout'],
        ).to(config.DEVICE)
        print(f"  {model}")

        train_loader = DataLoader(
            TensorDataset(X_train), batch_size=config.BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val), batch_size=config.BATCH_SIZE
        )

        print(f"  Training for {config.FINAL_EPOCHS} epochs...")
        _train_model(model, train_loader, config.FINAL_EPOCHS, best_params['lr'])

        print("  Evaluating...")
        fold_aurocs = _evaluate_fold(model, val_loader, fold_anom, fold_idx)
        mean_auroc  = np.mean(list(fold_aurocs.values()))
        print(f"  Fold {fold_idx} mean AUROC: {mean_auroc:.4f}")
        all_fold_aurocs.append(fold_aurocs)

    return all_fold_aurocs


# ==================================================================== #
#  Phase 3 — Final model on all data                                   #
# ==================================================================== #

def run_final_training(normal_files, best_params, n_feats):
    """
    Trains one model on ALL normal files using the best hyperparameters.
    Saves:
      - final_model.pt        (model weights + metadata)
      - imputer.joblib        (fitted IterativeImputer)
      - scaler.joblib         (fitted RobustScaler)
    These files are consumed by investigate.py.

    Returns the trained model.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: FINAL MODEL — ALL NORMAL DATA")
    print("=" * 60)

    hidden_layers = [
        best_params[f'layer_{i}_dim'] for i in range(best_params['n_layers'])
    ]

    final_imputer, final_scaler = process_and_fit_preprocessors(normal_files)
    save_preprocessors(final_imputer, final_scaler, config.OUTPUT_DIR)

    print("  Generating training windows...")
    X_all = process_files_to_windows(normal_files, final_imputer, final_scaler)
    print(f"  Total windows: {X_all.shape}")

    model = TunableAutoencoder(
        input_dim=n_feats,
        window_size=config.WINDOW_SIZE,
        hidden_layers=hidden_layers,
        dropout=best_params['dropout'],
    ).to(config.DEVICE)
    print(f"  {model}")

    train_loader = DataLoader(
        TensorDataset(X_all), batch_size=config.BATCH_SIZE, shuffle=True
    )

    print(f"  Training for {config.FINAL_EPOCHS} epochs...")
    _train_model(model, train_loader, config.FINAL_EPOCHS, best_params['lr'])

    # Save model checkpoint
    ckpt_path = config.OUTPUT_DIR / 'final_model.pt'
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'best_params':      best_params,
            'n_feats':          n_feats,
            'hidden_layers':    hidden_layers,
            'window_size':      config.WINDOW_SIZE,
        },
        ckpt_path,
    )
    print(f"  Model saved → {ckpt_path}")

    return model


# ==================================================================== #
#  Main                                                                 #
# ==================================================================== #

if __name__ == "__main__":
    print(f"Device     : {config.DEVICE}")
    print(f"Debug mode : {config.DEBUG_MODE}")
    print(f"Output dir : {config.OUTPUT_DIR}")

    normal_files, anomaly_files_dict = get_file_paths()

    # ---- Phase 1 ----
    best_params, n_feats = run_optuna_phase(normal_files)

    # ---- Phase 2 ----
    all_fold_aurocs = run_kfold_phase(
        normal_files, anomaly_files_dict, best_params, n_feats
    )

    # ---- Phase 3 ----
    run_final_training(normal_files, best_params, n_feats)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("FINAL CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    results_df = pd.DataFrame(all_fold_aurocs)
    results_df.index = [f'Fold_{i + 1}' for i in range(len(all_fold_aurocs))]

    summary = pd.concat(
        [
            results_df.mean().rename('Mean_AUROC'),
            results_df.max().rename('Max_AUROC'),
            results_df.std().rename('Std_Dev'),
        ],
        axis=1,
    )

    print(summary.to_string(float_format='{:.4f}'.format))
    overall = summary['Mean_AUROC'].mean()
    print(f"\nOVERALL MEAN SYSTEM AUROC: {overall:.4f}")

    summary.to_csv(config.OUTPUT_DIR / 'kfold_summary.csv')
    pd.DataFrame([best_params]).to_csv(
        config.OUTPUT_DIR / 'best_params.csv', index=False
    )

    print(f"\nAll outputs saved to {config.OUTPUT_DIR}")