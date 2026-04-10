import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
import pandas as pd
import config
from dataloader1 import load_data
from modelattncoder import AttentionAutoencoder
import os

def run_fold_analysis(fold_idx, model, val_loader, anom_dict, device):
    model.eval()
    print(f"\n--- FOLD {fold_idx} INTERNAL ANALYSIS ---")
    
    # 1. Baseline Normal Analysis
    norm_losses = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            recon, _ = model(x)
            loss = torch.mean((recon - x)**2, dim=[1, 2])
            norm_losses.extend(loss.cpu().numpy())
    
    mean_norm = np.mean(norm_losses)
    print(f"Normal Reconstruction Stability: {1 - mean_norm:.4%}")

    # 2. Comprehensive Class-by-Class Evaluation
    class_results = {}
    plt.figure(figsize=(10, 8))
    
    for cls, tensor in sorted(anom_dict.items()):
        anom_loader = DataLoader(TensorDataset(tensor), batch_size=512)
        anom_losses = []
        with torch.no_grad():
            for batch in anom_loader:
                x = batch[0].to(device)
                recon, _ = model(x)
                loss = torch.mean((recon - x)**2, dim=[1, 2])
                anom_losses.extend(loss.cpu().numpy())
        
        y_true = [0]*len(norm_losses) + [1]*len(anom_losses)
        y_scores = np.concatenate([norm_losses, anom_losses])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        class_results[cls] = roc_auc
        
        # Add to Plot
        plt.plot(fpr, tpr, label=f'Class {cls} (AUC = {roc_auc:.4f})')
        print(f"Class {cls} AUROC: {roc_auc:.4f}")

    # Format and Save Fold Plot
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - Fold {fold_idx}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(f'meeting_prep/fold_{fold_idx}_roc.png')
    plt.close()
            
    return class_results

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs('meeting_prep', exist_ok=True)

    # 1. Load Data
    X_train_full, X_val_raw, X_test_norm_raw, X_test_anom_dict, N_FEATS = load_data()
    X_normal_all = torch.cat([X_train_full, X_val_raw, X_test_norm_raw], dim=0)
    
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    final_stats = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_normal_all)):
        print(f"\nSTARTING FOLD {fold+1} / 4")
        
        X_train, X_val = X_normal_all[train_idx], X_normal_all[val_idx]
        train_loader = DataLoader(TensorDataset(X_train), batch_size=512, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val), batch_size=512)

        # Initialize Attention model with 4-min window
        model = AttentionAutoencoder(input_dim=N_FEATS, window_size=240, bottleneck_dim=64).to(config.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        
        # Training
        for epoch in range(15):
            model.train()
            total_loss = 0
            for batch in train_loader:
                x = batch[0].to(config.DEVICE)
                optimizer.zero_grad()
                recon, _ = model(x)
                loss = torch.mean((recon - x)**2)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Training Loss {total_loss/len(train_loader):.6f}")

        # Evaluate and Plot
        res = run_fold_analysis(fold+1, model, val_loader, X_test_anom_dict, config.DEVICE)
        final_stats.append(res)

    # 3. Final Aggregation and Export
    df_results = pd.DataFrame(final_stats)
    df_results.index = [f'Fold_{i+1}' for i in range(4)]
    
    summary = pd.concat([
        df_results.mean().rename('Mean_AUROC'), 
        df_results.max().rename('Max_AUROC'),
        df_results.std().rename('Std_Dev')
    ], axis=1)

    print("\n=== FINAL K-FOLD PERFORMANCE SUMMARY ===")
    print(summary)
    
    total_mean = summary['Mean_AUROC'].mean()
    print(f"\nOVERALL MEAN SYSTEM AUROC: {total_mean:.4f}")

    # Save to CSV for the report
    summary.to_csv('meeting_prep/kfold_final_results.csv')
    print("\nResults saved to meeting_prep/kfold_final_results.csv")