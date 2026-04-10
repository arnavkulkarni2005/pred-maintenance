import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

import config
from dataloader1 import load_data
from model import TunableAutoencoder

def objective(trial, X_train, X_val, input_dim):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []
    
    for i in range(n_layers):
        dim = trial.suggest_int(f'layer_{i}_dim', 16, 128)
        layers.append(dim)
        
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512]) 
    
    model = TunableAutoencoder(
        input_dim=input_dim, 
        window_size=config.WINDOW_SIZE, 
        hidden_layers=layers, 
        dropout=dropout
    ).to(config.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size)
    
    for epoch in range(config.TRIAL_EPOCHS):
        model.train()
        for batch in train_loader:
            x = batch[0].to(config.DEVICE)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(config.DEVICE)
                val_loss += criterion(model(x), x).item()
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            del model, optimizer, criterion
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
            
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    return val_loss

if __name__ == "__main__":
    print(f"Executing on device: {config.DEVICE}")
    
    # 1. Load Data (Anomalies are now a dict)
    X_train, X_val, X_test_norm, X_test_anom_dict, N_FEATS = load_data()
    
    # 2. Run Optuna
    print("\nStarting Optimization...")
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=config.PRUNER_STARTUP_TRIALS, 
        n_warmup_steps=config.PRUNER_WARMUP_STEPS
    )
    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(lambda trial: objective(trial, X_train, X_val, N_FEATS), n_trials=config.N_TRIALS) 
    print("\nBest params:", study.best_params)
    
    # 3. Train Best Model Fully
    best_layers = [study.best_params[f'layer_{i}_dim'] for i in range(study.best_params['n_layers'])]
    final_model = TunableAutoencoder(
        input_dim=N_FEATS,
        window_size=config.WINDOW_SIZE,
        hidden_layers=best_layers,
        dropout=study.best_params['dropout']
    ).to(config.DEVICE)
    
    best_batch = study.best_params.get('batch_size', 256)
    optimizer = optim.Adam(final_model.parameters(), lr=study.best_params['lr'])
    
    train_loader = DataLoader(TensorDataset(X_train), batch_size=best_batch, shuffle=True)
    
    print(f"Training Final Model for {config.FINAL_EPOCHS} epochs...")
    for epoch in range(config.FINAL_EPOCHS):
        final_model.train()
        for batch in train_loader:
            x = batch[0].to(config.DEVICE)
            optimizer.zero_grad()
            recon = final_model(x)
            loss = torch.mean((recon - x)**2)
            loss.backward()
            optimizer.step()
            
    # =======================================================
    # 4. DIAGNOSTIC EVALUATION (The "Truth Serum")
    # =======================================================
    print("\n" + "="*50)
    print("🔬 DIAGNOSTIC EVALUATION PHASE")
    print("="*50)
    final_model.eval()
    
    def get_eval_metrics(data):
        loader = DataLoader(TensorDataset(data), batch_size=256)
        losses = []
        all_x, all_recon = [], []
        
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(config.DEVICE)
                recon = final_model(x)
                
                # Window-level loss (for ROC)
                batch_loss = torch.mean((recon - x)**2, dim=[1, 2])
                losses.extend(batch_loss.cpu().numpy())
                
                all_x.append(x.cpu().numpy())
                all_recon.append(recon.cpu().numpy())
                
        return np.array(losses), np.concatenate(all_x), np.concatenate(all_recon)

    print("Evaluating Normal Test Data...")
    loss_normal, x_norm, recon_norm = get_eval_metrics(X_test_norm)
    print(f"Normal Baseline MSE: {np.mean(loss_normal):.4f}")

    all_anomaly_losses = []
    
    for cls, anom_data in sorted(X_test_anom_dict.items()):
        print(f"\nEvaluating Class {cls}...")
        loss_anom, x_anom, recon_anom = get_eval_metrics(anom_data)
        all_anomaly_losses.extend(loss_anom)
        
        # Calculate Class AUROC
        y_true_cls = [0]*len(loss_normal) + [1]*len(loss_anom)
        y_scores_cls = np.concatenate([loss_normal, loss_anom])
        fpr_cls, tpr_cls, _ = roc_curve(y_true_cls, y_scores_cls)
        roc_auc_cls = auc(fpr_cls, tpr_cls)
        
        print(f"  -> AUROC: {roc_auc_cls:.4f} ({len(loss_anom)} windows)")
        
        # Calculate Feature-Wise MSE (Which sensor caused the anomaly?)
        feature_mse = np.mean((recon_anom - x_anom)**2, axis=(0, 1))
        
        print("  -> Top 3 error-driving sensors:")
        top_indices = np.argsort(-feature_mse)[:3]
        for idx in top_indices:
            print(f"     {config.TARGET_SENSORS[idx]}: MSE = {feature_mse[idx]:.4f}")

    # Overall AUROC
    y_true_all = [0]*len(loss_normal) + [1]*len(all_anomaly_losses)
    y_scores_all = np.concatenate([loss_normal, all_anomaly_losses])
    fpr_all, tpr_all, _ = roc_curve(y_true_all, y_scores_all)
    roc_auc_all = auc(fpr_all, tpr_all)
    
    print("\n" + "="*50)
    print(f"🏆 FINAL OVERALL AUROC: {roc_auc_all:.4f}")
    print("="*50)
    
    # Save Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_all, tpr_all, color='darkorange', lw=2, label=f'Overall ROC (area = {roc_auc_all:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Diagnostic Autoencoder Detection Performance')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    save_path = config.OUTPUT_DIR / "diagnostic_roc.png"
    plt.savefig(save_path)
    print(f"ROC Curve saved to {save_path}")