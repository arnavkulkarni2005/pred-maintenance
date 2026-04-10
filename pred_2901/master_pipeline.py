import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, train_test_split
import optuna
import os

import config
from strict_dataloader import get_file_paths, process_and_fit_preprocessors, process_files_to_windows

# --- MODEL DEFINITION ---
class TunableAutoencoder(nn.Module):
    def __init__(self, input_dim, window_size, hidden_layers=[64, 32], dropout=0.0):
        super().__init__()
        
        self.input_flat = window_size * input_dim
        
        # Build Encoder
        encoder_layers = []
        in_dim = self.input_flat
        
        for h_dim in hidden_layers:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build Decoder (Mirrored)
        decoder_layers = []
        hidden_layers_reversed = hidden_layers[:-1][::-1] + [self.input_flat]
        
        for h_dim in hidden_layers_reversed:
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            if h_dim != self.input_flat: 
                decoder_layers.append(nn.ReLU())
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        b, w, f = x.shape
        x_flat = x.view(b, -1)
        
        latent = self.encoder(x_flat)
        recon_flat = self.decoder(latent)
        
        recon = recon_flat.view(b, w, f)
        return recon

# --- EVALUATION LOGIC ---
def evaluate_model(model, val_loader, anom_dict, device, fold_idx):
    model.eval()
    norm_losses = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            recon = model(x)
            loss = torch.mean((recon - x)**2, dim=[1, 2])
            norm_losses.extend(loss.cpu().numpy())
            
    class_results = {}
    plt.figure(figsize=(10, 8))
    
    for cls, anom_tensor in sorted(anom_dict.items()):
        if len(anom_tensor) == 0:
            continue
            
        anom_loader = DataLoader(TensorDataset(anom_tensor), batch_size=512)
        anom_losses = []
        with torch.no_grad():
            for batch in anom_loader:
                x = batch[0].to(device)
                recon = model(x)
                loss = torch.mean((recon - x)**2, dim=[1, 2])
                anom_losses.extend(loss.cpu().numpy())
                
        y_true = [0]*len(norm_losses) + [1]*len(anom_losses)
        y_scores = np.concatenate([norm_losses, anom_losses])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        class_results[cls] = roc_auc
        
        plt.plot(fpr, tpr, label=f'Class {cls} (AUC = {roc_auc:.4f})')
        print(f"  Class {cls} AUROC: {roc_auc:.4f}")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Strict File-Split ROC (MLP) - Fold {fold_idx}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(f'meeting_prep/strict_mlp_fold_{fold_idx}_roc.png')
    plt.close()
    
    return class_results

# --- PIPELINE EXECUTOR ---
if __name__ == "__main__":
    os.makedirs('meeting_prep', exist_ok=True)
    N_FEATS = len(config.TARGET_SENSORS)
    normal_files, anomaly_files_dict = get_file_paths()
    
    print("\n--- PHASE 1: OPTUNA HYPERPARAMETER TUNING (MLP) ---")
    opt_train_files, opt_val_files = train_test_split(normal_files, test_size=0.2, random_state=42)
    opt_imputer, opt_scaler = process_and_fit_preprocessors(opt_train_files)
    
    print("Generating Optuna Window Tensors...")
    X_train_opt = process_files_to_windows(opt_train_files, opt_imputer, opt_scaler, config.WINDOW_SIZE, config.STRIDE)
    X_val_opt = process_files_to_windows(opt_val_files, opt_imputer, opt_scaler, config.WINDOW_SIZE, config.STRIDE)
    
    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 2)
        hidden_layers = []
        for i in range(n_layers):
            dim = trial.suggest_int(f'layer_{i}_dim', 32, 128, step=16)
            hidden_layers.append(dim)
            
        dropout = trial.suggest_float('dropout', 0.0, 0.2)
        lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        
        model = TunableAutoencoder(N_FEATS, config.WINDOW_SIZE, hidden_layers, dropout).to(config.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        train_loader = DataLoader(TensorDataset(X_train_opt), batch_size=512, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_opt), batch_size=512)
        
        for epoch in range(10):
            # Train for one epoch
            model.train()
            for batch in train_loader:
                x = batch[0].to(config.DEVICE)
                optimizer.zero_grad()
                recon = model(x)
                loss = torch.mean((recon - x)**2)
                loss.backward()
                optimizer.step()
                
            # Evaluate Validation MSE for this epoch
            model.eval()
            val_mse = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(config.DEVICE)
                    recon = model(x)
                    val_mse += torch.mean((recon - x)**2).item()
            val_mse /= len(val_loader)
            
            # --- THE PRUNER LOGIC ---
            trial.report(val_mse, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
        return val_mse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    print(f"\nOptimal Unsupervised Params Found: {best_params}")

    print("\n--- PHASE 2: STRICT 4-FOLD EVALUATION (MLP) ---")
    
    best_hidden_layers = [best_params[f'layer_{i}_dim'] for i in range(best_params['n_layers'])]
    
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    normal_files_array = np.array(normal_files)
    all_fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(normal_files_array)):
        print(f"\n================ FOLD {fold+1} / 4 ================")
        train_files = normal_files_array[train_idx].tolist()
        val_files = normal_files_array[val_idx].tolist()
        
        fold_imputer, fold_scaler = process_and_fit_preprocessors(train_files)
        
        print("Generating Normal Train/Val Tensors...")
        X_train = process_files_to_windows(train_files, fold_imputer, fold_scaler, config.WINDOW_SIZE, config.STRIDE)
        X_val = process_files_to_windows(val_files, fold_imputer, fold_scaler, config.WINDOW_SIZE, config.STRIDE)
        
        print("Processing Anomaly Tensors via Fold Preprocessor...")
        fold_anom_tensors = {}
        for cls, f_list in anomaly_files_dict.items():
            fold_anom_tensors[cls] = process_files_to_windows(f_list, fold_imputer, fold_scaler, config.WINDOW_SIZE, config.STRIDE)

        model = TunableAutoencoder(
            input_dim=N_FEATS, 
            window_size=config.WINDOW_SIZE, 
            hidden_layers=best_hidden_layers,
            dropout=best_params['dropout']
        ).to(config.DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
        train_loader = DataLoader(TensorDataset(X_train), batch_size=512, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val), batch_size=512)
        
        print("Training MLP Autoencoder...")
        for epoch in range(15):
            model.train()
            for batch in train_loader:
                x = batch[0].to(config.DEVICE)
                optimizer.zero_grad()
                recon = model(x)
                loss = torch.mean((recon - x)**2)
                loss.backward()
                optimizer.step()
                
        print("Evaluating Anomalies and Saving Plots...")
        fold_aurocs = evaluate_model(model, val_loader, fold_anom_tensors, config.DEVICE, fold+1)
        all_fold_results.append(fold_aurocs)

    final_df = pd.DataFrame(all_fold_results)
    final_df.index = [f'Fold_{i+1}' for i in range(4)]
    
    summary = pd.concat([
        final_df.mean().rename('Mean_AUROC'),
        final_df.max().rename('Max_AUROC'),
        final_df.std().rename('Std_Dev')
    ], axis=1)
    
    print("\n" + "="*50)
    print("FINAL STRICT NO-LEAKAGE SYSTEM PERFORMANCE (MLP)")
    print("="*50)
    print(summary)
    print(f"\nOVERALL MEAN SYSTEM AUROC: {summary['Mean_AUROC'].mean():.4f}")
    
    summary.to_csv('meeting_prep/strict_mlp_final_aurocs.csv')
    print("\nResults and all ROC plots successfully saved to meeting_prep/")