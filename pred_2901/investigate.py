import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import config
from dataloader1 import load_data
from modelattncoder import AttentionAutoencoder

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

def analyze_single_file(model, file_path, imputer, scaler, device):
    print(f"\nAnalyzing File: {file_path.name}")
    
    # 1. Load and Clean exactly one file
    df = pd.read_parquet(file_path)
    for sensor in config.TARGET_SENSORS:
        if sensor not in df.columns:
            df[sensor] = np.nan
            
    df = df[config.TARGET_SENSORS]
    imputed_vals = imputer.transform(df)
    df_clean = pd.DataFrame(imputed_vals, columns=config.TARGET_SENSORS).ffill().fillna(0.0)
    
    # 2. Scale
    scaled_vals = scaler.transform(df_clean.values)
    scaled_vals = np.clip(scaled_vals, -15.0, 15.0)
    
    # 3. Windowing (Chronological, NO shuffle)
    windows = []
    for i in range(0, len(scaled_vals) - config.WINDOW_SIZE, config.STRIDE):
        windows.append(scaled_vals[i:i+config.WINDOW_SIZE])
        
    X_tensor = torch.tensor(np.array(windows, dtype=np.float32)).to(device)
    
    # 4. Get MSE over time
    model.eval()
    losses = []
    with torch.no_grad():
        recon, _ = model(X_tensor)
        loss = torch.mean((recon - X_tensor)**2, dim=[1, 2])
        losses = loss.cpu().numpy()
        
    # 5. Plot the Proof
    plt.figure(figsize=(12, 8))
    
    # Plot Raw Pressure (P-TPT is usually index 0 or 1 depending on your config)
    # We will just plot the first feature of the unscaled data for visual reference
    plt.subplot(2, 1, 1)
    plt.plot(df_clean.iloc[:, 0].values, color='blue', alpha=0.7)
    plt.title(f'Raw Sensor Reading: {config.TARGET_SENSORS[0]} (Notice where the fault actually begins)')
    plt.ylabel('Raw Value')
    plt.grid(True, alpha=0.3)
    
    # Plot Model MSE
    plt.subplot(2, 1, 2)
    # Create an x-axis that matches the original dataframe length based on strides
    time_axis = np.arange(len(losses)) * config.STRIDE + config.WINDOW_SIZE
    plt.plot(time_axis, losses, color='red', linewidth=2)
    plt.title('Autoencoder Reconstruction Error (MSE)')
    plt.ylabel('MSE')
    plt.xlabel('Timestep')
    plt.grid(True, alpha=0.3)
    
    # Draw a line for the "Normal Baseline" threshold (e.g., MSE = 1.0)
    plt.axhline(y=1.0, color='black', linestyle='--', label='Theoretical Normal Threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('meeting_prep/leak_investigation.png')
    print("Saved chronological plot to meeting_prep/leak_investigation.png")

if __name__ == "__main__":
    # We need the dataloader just to get the fitted imputer and scaler
    print("Loading data to fit global transforms...")
    X_train_full, X_val_raw, X_test_norm_raw, X_test_anom_dict, N_FEATS = load_data()
    
    # Mock a quick training run to get a functioning model
    print("Training rapid diagnostic model...")
    model = AttentionAutoencoder(input_dim=N_FEATS, window_size=240, bottleneck_dim=64).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(TensorDataset(X_train_full), batch_size=512, shuffle=True)
    
    model.train()
    for epoch in range(3): # Just 3 epochs is enough to drop the normal MSE
        for batch in train_loader:
            x = batch[0].to(config.DEVICE)
            optimizer.zero_grad()
            recon, _ = model(x)
            loss = torch.mean((recon - x)**2)
            loss.backward()
            optimizer.step()
            
    # We need to hack into the dataloader namespace to grab the global imputer and scaler 
    # (In a clean codebase, these would be returned or saved to disk)
    import dataloader1
    import sys
    
    # Grab the first file from Class 3
    class_3_path = config.DATASET_PATH / '3'
    test_file = list(class_3_path.rglob("*.parquet"))[0]
    
    # Note: For this to run perfectly, we need access to your fitted imputer and scaler.
    # I am assuming your dataloader1.py exposes `global_imputer` and `scaler` globally, 
    # or you can temporarily add `return global_imputer, scaler` to load_data().
    
    print("\n--- INITIATING LEAK INVESTIGATION ---")
    print("If the MSE graph spikes to 50+ at timestep 0, the model is cheating.")
    print("If the MSE stays near 0 until the wave gets chaotic, the physics are sound.")
    
    # Assuming you temporarily modified load_data to return the imputer and scaler:
    # imputer, scaler = ... 
    # analyze_single_file(model, test_file, imputer, scaler, config.DEVICE)