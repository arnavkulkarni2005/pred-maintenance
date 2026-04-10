import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import optuna
import pandas as pd
import os
from sklearn.model_selection import KFold

import config
from dataloader1 import load_data
from modelattncoder import AttentionAutoencoder

def objective(trial):
    # 1. Hyperparameters to optimize
    bottleneck_dim = trial.suggest_int("bottleneck_dim", 32, 160, step=16)
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.2)
    
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    fold_mses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_normal_all)):
        X_train_fold, X_val_fold = X_normal_all[train_idx], X_normal_all[val_idx]
        train_loader = DataLoader(TensorDataset(X_train_fold), batch_size=512, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_fold), batch_size=512)

        model = AttentionAutoencoder(
            input_dim=N_FEATS, 
            window_size=240, 
            bottleneck_dim=bottleneck_dim,
            dropout=dropout
        ).to(config.DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training loop
        model.train()
        for epoch in range(10): # Efficient tuning
            for batch in train_loader:
                x = batch[0].to(config.DEVICE)
                optimizer.zero_grad()
                recon, _ = model(x)
                loss = torch.mean((recon - x)**2)
                loss.backward()
                optimizer.step()

        # Evaluate on Validation MSE (The Unsupervised Metric)
        model.eval()
        val_mse = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(config.DEVICE)
                recon, _ = model(x)
                val_mse += torch.mean((recon - x)**2).item()
        
        fold_mses.append(val_mse / len(val_loader))
        
    # We want to MINIMIZE the reconstruction error of normal data
    return np.mean(fold_mses)

if __name__ == "__main__":
    os.makedirs('meeting_prep', exist_ok=True)
    
    # Load Data
    X_train_full, X_val_raw, X_test_norm_raw, X_test_anom_dict, N_FEATS = load_data()
    X_normal_all = torch.cat([X_train_full, X_val_raw, X_test_norm_raw], dim=0)

    print("--- OPTUNA: MINIMIZING NORMAL RECONSTRUCTION ERROR (MSE) ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print("\nBest Unsupervised Params:")
    print(study.best_params)
    
    # Save results
    best_df = pd.DataFrame([study.best_params])
    best_df.to_csv('meeting_prep/best_unsupervised_params.csv', index=False)