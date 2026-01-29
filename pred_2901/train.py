import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path

# Import our modules
from dataloader import load_data
from model import TunableAutoencoder

# CONFIG
DATASET_PATH = Path('/home/kulkarni/projects/pred_maintenance/dataset')
OUTPUT_DIR = Path('/home/kulkarni/projects/pred_maintenance/meeting_prep')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20 # Keep small for Optuna speed

def objective(trial):
    # 1. Suggest Hyperparameters
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []
    current_dim = 256 # Starting abstract dim
    
    for i in range(n_layers):
        dim = trial.suggest_int(f'layer_{i}_dim', 16, 128)
        layers.append(dim)
        
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    
    # 2. Build Model
    model = TunableAutoencoder(
        input_dim=N_FEATS, 
        window_size=120, 
        hidden_layers=layers, 
        dropout=dropout
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 3. Train Loop (Quick)
    train_loader = DataLoader(TensorDataset(X_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=128)
    
    for epoch in range(5): # Short epochs for pruning
        model.train()
        for batch in train_loader:
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(DEVICE)
                val_loss += criterion(model(x), x).item()
        
        # Pruning (Stop bad trials early)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_loss

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data
    X_train, X_val, X_test_norm, X_test_anom, N_FEATS = load_data(DATASET_PATH)
    
    # 2. Run Optuna
    print("Starting Optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10) # Set to 20-50 for real run
    
    print("Best params:", study.best_params)
    
    # 3. Train Best Model Fully
    best_layers = [study.best_params[f'layer_{i}_dim'] for i in range(study.best_params['n_layers'])]
    final_model = TunableAutoencoder(
        input_dim=N_FEATS,
        window_size=120,
        hidden_layers=best_layers,
        dropout=study.best_params['dropout']
    ).to(DEVICE)
    
    optimizer = optim.Adam(final_model.parameters(), lr=study.best_params['lr'])
    criterion = nn.MSELoss(reduction='none') # Keep per-element loss
    
    train_loader = DataLoader(TensorDataset(X_train), batch_size=128, shuffle=True)
    
    print("Training Final Model...")
    for epoch in range(20):
        final_model.train()
        for batch in train_loader:
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            recon = final_model(x)
            loss = torch.mean((recon - x)**2) # Scalar mean for optimization
            loss.backward()
            optimizer.step()
            
    # 4. Evaluation (ROC Curve)
    print("Evaluating...")
    final_model.eval()
    
    def get_losses(data):
        loader = DataLoader(TensorDataset(data), batch_size=128)
        losses = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(DEVICE)
                recon = final_model(x)
                # Calculate MSE per window [Batch]
                # Mean over features and time
                batch_loss = torch.mean((recon - x)**2, dim=[1, 2])
                losses.extend(batch_loss.cpu().numpy())
        return losses

    loss_normal = get_losses(X_test_norm)
    loss_anom = get_losses(X_test_anom)
    
    # Create Labels (0=Normal, 1=Anomaly)
    y_true = [0]*len(loss_normal) + [1]*len(loss_anom)
    y_scores = loss_normal + loss_anom
    
    # Calculate ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Baseline Autoencoder Detection Performance')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    save_path = OUTPUT_DIR / "final_baseline_roc.png"
    plt.savefig(save_path)
    print(f"ROC Curve saved to {save_path}")
    print(f"FINAL AUROC: {roc_auc:.4f}")