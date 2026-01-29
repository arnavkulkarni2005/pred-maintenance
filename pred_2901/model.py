import torch
import torch.nn as nn

class TunableAutoencoder(nn.Module):
    def __init__(self, input_dim, window_size, hidden_layers=[64, 32], dropout=0.0):
        super().__init__()
        
        # Flatten input [B, W, F] -> [B, W*F]
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
            if h_dim != self.input_flat: # No ReLU/Dropout on final layer
                decoder_layers.append(nn.ReLU())
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # x shape: [Batch, Window, Features]
        b, w, f = x.shape
        x_flat = x.view(b, -1)
        
        latent = self.encoder(x_flat)
        recon_flat = self.decoder(latent)
        
        recon = recon_flat.view(b, w, f)
        return recon