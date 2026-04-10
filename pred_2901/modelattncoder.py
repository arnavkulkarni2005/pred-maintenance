import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionAutoencoder(nn.Module):
    def __init__(self, input_dim, window_size, bottleneck_dim=64, dropout=0.05):
        super().__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        
        # 🧠 THE NOVELTY: Temporal Self-Attention
        # This allows the model to "focus" on specific time-steps in the 4-min window
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(window_size * input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, bottleneck_dim), # Tightened Bottleneck
            nn.LeakyReLU(0.2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, window_size * input_dim)
        )

    def attention_layer(self, x):
        # x shape: [batch, window, features]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Attention scores: How much does each timestep relate to others?
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.input_dim).float())
        weights = F.softmax(scores, dim=-1)
        
        # Re-weighted sequence
        context = torch.matmul(weights, V)
        return context

    def forward(self, x):
        b, w, f = x.shape
        
        # 1. Apply Attention
        attn_x = self.attention_layer(x)
        
        # 2. Flatten & Encode
        flat_x = attn_x.view(b, -1)
        latent = self.encoder(flat_x)
        
        # 3. Decode & Reshape
        recon_flat = self.decoder(latent)
        recon = recon_flat.view(b, w, f)
        
        return recon, latent