"""
model.py

Defines TunableAutoencoder: a plain MLP autoencoder for
unsupervised time-series anomaly detection.

The model flattens each [B, W, F] window into [B, W*F], passes it
through a symmetric encoder–decoder, then reshapes back to [B, W, F].
Anomaly score = per-window MSE between input and reconstruction.

Usage example
-------------
    from model import TunableAutoencoder

    model = TunableAutoencoder(
        input_dim=8,
        window_size=240,
        hidden_layers=[256, 64],   # 256 hidden, 64 bottleneck
        dropout=0.05,
    )
    recon = model(x)               # [B, 240, 8]
    scores = model.reconstruction_error(x)  # [B]
"""

import torch
import torch.nn as nn


class TunableAutoencoder(nn.Module):
    """
    MLP autoencoder with a configurable symmetric architecture.

    Parameters
    ----------
    input_dim     : number of sensor features (F)
    window_size   : timesteps per window (W)
    hidden_layers : list of layer widths from input to bottleneck.
                    e.g. [128, 64] → encoder: W*F→128→64
                                   → decoder: 64→128→W*F
                    The last value is the bottleneck dimension.
    dropout       : dropout probability applied after every hidden
                    ReLU (0.0 disables dropout entirely)
    """

    def __init__(self, input_dim, window_size, hidden_layers=None, dropout=0.0):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [64]

        self.input_dim    = input_dim
        self.window_size  = window_size
        self.input_flat   = window_size * input_dim
        self.bottleneck   = hidden_layers[-1]

        # ---- Encoder ------------------------------------------------
        # input_flat → h0 → h1 → … → bottleneck
        enc_layers = []
        in_dim = self.input_flat
        for h_dim in hidden_layers:
            enc_layers.append(nn.Linear(in_dim, h_dim))
            enc_layers.append(nn.ReLU())
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            in_dim = h_dim                  # in_dim = bottleneck after loop

        self.encoder = nn.Sequential(*enc_layers)

        # ---- Decoder ------------------------------------------------
        # bottleneck → h_{n-2} → … → h0 → input_flat
        # Mirrors the encoder but drops the activation on the final layer.
        #
        # Example: hidden_layers = [256, 128, 64]
        #   encoder dims: input_flat→256→128→64
        #   decoder dims: 64→128→256→input_flat
        #
        # reversed(hidden_layers[:-1]) gives the "mirror" hidden dims.
        # Appending input_flat gives the final output projection.
        dec_dims   = list(reversed(hidden_layers[:-1])) + [self.input_flat]
        dec_layers = []
        for i, h_dim in enumerate(dec_dims):
            dec_layers.append(nn.Linear(in_dim, h_dim))
            is_last = (i == len(dec_dims) - 1)
            if not is_last:
                dec_layers.append(nn.ReLU())
                if dropout > 0:
                    dec_layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.decoder = nn.Sequential(*dec_layers)

    # ---------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x     : [B, W, F]
        return: [B, W, F]  (reconstructed window)
        """
        b, w, f = x.shape
        latent = self.encoder(x.view(b, -1))
        recon  = self.decoder(latent).view(b, w, f)
        return recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the bottleneck latent vector: [B, bottleneck]."""
        b, w, f = x.shape
        return self.encoder(x.view(b, -1))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-window mean-squared reconstruction error.
        x      : [B, W, F]
        return : [B]   (one scalar score per window)
        """
        recon = self(x)
        return torch.mean((recon - x) ** 2, dim=[1, 2])

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"TunableAutoencoder("
            f"input_flat={self.input_flat}, "
            f"bottleneck={self.bottleneck}, "
            f"params={n_params:,})"
        )