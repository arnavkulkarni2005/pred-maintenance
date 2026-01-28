import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # pip install tqdm

# ================= CONFIGURATION =================
DATASET_PATH = Path('/home/kulkarni/projects/pred_maintenance/dataset')
OUTPUT_DIR = Path('/home/kulkarni/projects/pred_maintenance/outlook')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CLASS = 0       
EPOCHS = 50           # Reduced slightly for CPU speed
BATCH_SIZE = 256      
TRIALS_PER_DIM = 3    
POINTS_PER_FILE = 2000 # Reduced to 2000 to save CPU RAM/Time

# FORCE CPU
DEVICE = torch.device("cpu")
print(f"Forcefully running on: {DEVICE}")
# =================================================

def load_full_dataset():
    all_data = []
    class_path = DATASET_PATH / str(TRAIN_CLASS)
    
    files = list(class_path.rglob("*.parquet"))
    print(f"Found {len(files)} files in Class {TRAIN_CLASS}. Loading data...")
    
    valid_cols = None

    for f in tqdm(files, desc="Loading Parquet Files"):
        try:
            df = pd.read_parquet(f)
            
            # Select Sensors
            cols = [c for c in df.columns if ('P-' in c or 'T-' in c or 'Q' in c) 
                    and 'ESTADO' not in c and 'state' not in c.lower()]
            
            if not cols: continue
            
            # Drop dead sensors
            good_cols = [c for c in cols if df[c].notna().any()]
            if not good_cols: continue
            
            df_clean = df[good_cols].ffill().bfill().dropna()
            if df_clean.empty: continue

            # Schema Check
            if valid_cols is None:
                valid_cols = good_cols
            else:
                missing = [c for c in valid_cols if c not in df_clean.columns]
                if missing: continue
                df_clean = df_clean[valid_cols]

            # SAMPLING
            n = min(len(df_clean), POINTS_PER_FILE)
            sample = df_clean.sample(n=n, replace=False)
            all_data.append(sample)
            
        except:
            continue

    if not all_data: raise ValueError("No data found!")
    
    print("Concatenating data...")
    full_df = pd.concat(all_data, ignore_index=True)
    print(f"Final Dataset Size: {len(full_df)} rows, {full_df.shape[1]} columns")
    
    scaler = StandardScaler()
    X = scaler.fit_transform(full_df)
    return torch.tensor(X, dtype=torch.float32), X.shape[1]

# --- MODEL ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- TRAINER ---
def train_trial(train_loader, val_loader, input_dim, hidden_dim):
    # Ensure model is initialized on CPU
    model = Autoencoder(input_dim, hidden_dim).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            x_batch = batch[0].to(DEVICE) # Explicitly move data to CPU
            
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, x_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_val = batch[0].to(DEVICE)
                val_loss += criterion(model(x_val), x_val).item()
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
    return best_val_loss

if __name__ == "__main__":
    X, input_dim = load_full_dataset()
    
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    
    train_ds = TensorDataset(X_train)
    val_ds = TensorDataset(X_val)
    
    # Num_workers=0 is safer for CPU scripts to avoid multiprocessing overhead
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    
    results = {}
    print(f"\nStarting CPU Analysis on {DEVICE}...")
    
    for h_dim in range(1, input_dim):
        print(f"Testing Latent Dimension: {h_dim}...")
        trial_losses = []
        
        for t in range(TRIALS_PER_DIM):
            loss = train_trial(train_loader, val_loader, input_dim, h_dim)
            trial_losses.append(loss)
        
        best_loss = min(trial_losses)
        results[h_dim] = best_loss
        print(f"  -> Best MSE: {best_loss:.5f}")

    # Plot
    dims = list(results.keys())
    losses = list(results.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(dims, losses, marker='o', linewidth=2, color='#27ae60') # Green for CPU
    plt.xlabel("Number of Latent Variables")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.title(f"Full-Scale Complexity Analysis (CPU Mode)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(dims)
    
    save_path = OUTPUT_DIR / "autoencoder_full_scale_cpu.png"
    plt.savefig(save_path)
    print(f"Saved CPU analysis to {save_path}")