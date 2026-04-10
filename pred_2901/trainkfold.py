import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
import pandas as pd

import config
from dataloader1 import load_data
from model import TunableAutoencoder

def evaluate_model(model, data_loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            recon = model(x)
            # Calculate window-level MSE
            batch_loss = torch.mean((recon - x)**2, dim=[1, 2])
            losses.extend(batch_loss.cpu().numpy())
    return np.array(losses)

if __name__ == "__main__":
    print(f"🚀 Initializing 4-Fold Cross-Validation on {config.DEVICE}")
    
    # 1. Load the full dataset (Anomalies remain constant, Normal data is used for K-Fold)
    # Note: Ensure your load_data returns the FULL normal set for K-Folding
    X_train_full, X_val_raw, X_test_norm_raw, X_test_anom_dict, N_FEATS = load_data()
    
    # Combine all normal data for folding (roughly 200k windows total)
    X_normal_all = torch.cat([X_train_full, X_val_raw, X_test_norm_raw], dim=0)
    
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    fold_results = []
    
    # Architecture from your best Optuna run
    BEST_PARAMS = {
        'n_layers': 1,
        'layer_0_dim': 123,
        'dropout': 0.0057,
        'lr': 0.0012,
        'batch_size': 512
    }

    print("\n" + "="*50)
    print(f"🏗️ ARCHITECTURE: 1-Layer MLP Autoencoder")
    print(f"   Input: {N_FEATS * config.WINDOW_SIZE} -> Bottleneck: {BEST_PARAMS['layer_0_dim']} -> Output")
    print("="*50)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_normal_all)):
        print(f"\n--- 🌀 STARTING FOLD {fold + 1}/4 ---")
        
        # Split data
        X_train = X_normal_all[train_idx]
        X_val = X_normal_all[val_idx]
        
        train_loader = DataLoader(TensorDataset(X_train), batch_size=BEST_PARAMS['batch_size'], shuffle=True)
        
        # Re-initialize model for each fold
        model = TunableAutoencoder(
            input_dim=N_FEATS,
            window_size=config.WINDOW_SIZE,
            hidden_layers=[BEST_PARAMS['layer_0_dim']],
            dropout=BEST_PARAMS['dropout']
        ).to(config.DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=BEST_PARAMS['lr'])
        
        # Train for 25 epochs
        for epoch in range(config.FINAL_EPOCHS):
            model.train()
            for batch in train_loader:
                x = batch[0].to(config.DEVICE)
                optimizer.zero_grad()
                recon = model(x)
                loss = torch.mean((recon - x)**2)
                loss.backward()
                optimizer.step()
        
        # Evaluate this fold
        model.eval()
        loss_norm_val = evaluate_model(model, DataLoader(TensorDataset(X_val), batch_size=256), config.DEVICE)
        
        current_fold_aurocs = {}
        for cls, anom_tensor in X_test_anom_dict.items():
            loss_anom = evaluate_model(model, DataLoader(TensorDataset(anom_tensor), batch_size=256), config.DEVICE)
            
            y_true = [0]*len(loss_norm_val) + [1]*len(loss_anom)
            y_scores = np.concatenate([loss_norm_val, loss_anom])
            current_fold_aurocs[cls] = auc(*roc_curve(y_true, y_scores)[:2])
            
        fold_results.append(current_fold_aurocs)
        print(f"✅ Fold {fold+1} complete. Avg AUROC: {np.mean(list(current_fold_aurocs.values())):.4f}")

    # =======================================================
    # 5. FINAL TABLE GENERATION
    # =======================================================
    print("\n" + "="*60)
    print("📊 FINAL CROSS-VALIDATION SUMMARY TABLE")
    print("="*60)
    
    results_df = pd.DataFrame(fold_results)
    results_df.index = [f'Fold {i+1}' for i in range(4)]
    
    # Calculate Mean and Max across folds
    stats_df = pd.DataFrame({
        'Mean AUROC': results_df.mean(),
        'Max AUROC': results_df.max()
    })
    
    final_table = pd.concat([results_df.T, stats_df], axis=1)
    print(final_table.to_string(formatters={col: "{:,.4f}".format for col in final_table.columns}))
    
    # Overall summary
    print("\n" + "="*60)
    print(f"🏆 TOTAL CROSS-VAL MEAN AUROC: {stats_df['Mean AUROC'].mean():.4f}")
    print("="*60)